"""SQLite state machine.

Everything about resumability lives here. Rules that must not be broken:

* Status transitions and the artifact row that justifies them are written in
  one transaction. A stage that finishes its work but dies before committing
  simply re-runs, which is why every stage overwrites its own output rather
  than appending.
* Blobs never go in the database, only paths and hashes.
* Errors are classified explicitly as retryable or permanent. Blanket-retrying
  a members-only video wastes hours; blanket-retrying a bot-check makes the
  block worse.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator

SCHEMA_DIR_NAME = "migrations"

# --- state machine ---------------------------------------------------------

NEW = "new"
FETCHED = "fetched"
TRANSCRIBED = "transcribed"
NORMALIZED = "normalized"
SUMMARIZED = "summarized"
DONE = "done"
FAILED = "failed"
ABANDONED = "abandoned"

# stage -> (status required to start, status written on success)
STAGES: dict[str, tuple[str, str]] = {
    "fetch": (NEW, FETCHED),
    "transcribe": (FETCHED, TRANSCRIBED),
    "normalize": (TRANSCRIBED, NORMALIZED),
    "summarize": (NORMALIZED, SUMMARIZED),
    "publish": (SUMMARIZED, DONE),
}
STAGE_ORDER = list(STAGES)

RETRYABLE = "retryable"
PERMANENT = "permanent"


class StageError(Exception):
    """Raised by a stage to signal a classified failure."""

    def __init__(self, message: str, error_class: str = RETRYABLE):
        super().__init__(message)
        self.error_class = error_class


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


# --- connection ------------------------------------------------------------


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def split_statements(sql: str) -> list[str]:
    """Split a migration file into individual statements.

    We cannot use executescript(): it issues an implicit COMMIT before running,
    which silently breaks out of our explicit transaction and leaves the schema
    applied but unrecorded. Accumulate lines until sqlite says the statement is
    complete — the stdlib's own definition of a statement boundary, so it is
    not fooled by semicolons inside string literals.
    """
    statements: list[str] = []
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                statements.append(buffer)
            buffer = ""
    if buffer.strip():
        statements.append(buffer)
    return statements


def migrate(conn: sqlite3.Connection, migrations_dir: Path) -> list[str]:
    """Apply numbered .sql files not yet recorded. Returns names applied.

    Each file lands with its schema_migrations row in one transaction, so a
    crash mid-migration re-runs the whole file rather than half of it.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "  name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}
    fresh: list[str] = []
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        if sql_file.name in applied:
            continue
        with transaction(conn):
            for statement in split_statements(sql_file.read_text(encoding="utf-8")):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                (sql_file.name, now_iso()),
            )
        fresh.append(sql_file.name)
    return fresh


def open_db(db_path: Path, migrations_dir: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    migrate(conn, migrations_dir)
    return conn


# --- queue -----------------------------------------------------------------


def claim_queue(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Videos ready for work, oldest published first.

    A `failed` video is only eligible once its backoff has elapsed. `abandoned`
    is never returned; it surfaces in `status` and waits for a human.
    """
    return list(
        conn.execute(
            """
            SELECT v.* FROM videos v
            WHERE v.status NOT IN (?, ?)
              AND (
                v.status <> ?
                OR COALESCE((
                     SELECT MAX(next_retry_at) FROM stage_runs r
                     WHERE r.video_id = v.id
                   ), '') <= ?
              )
            ORDER BY v.published_at ASC
            LIMIT ?
            """,
            (DONE, ABANDONED, FAILED, now_iso(), limit),
        )
    )


def next_stage_for(conn: sqlite3.Connection, video_id: str) -> str | None:
    """Which stage should run next for this video, if any."""
    row = conn.execute("SELECT status FROM videos WHERE id = ?", (video_id,)).fetchone()
    if row is None:
        return None
    status = row["status"]
    if status in (DONE, ABANDONED):
        return None
    if status == FAILED:
        # Resume at the stage whose precondition the last good status satisfies.
        status = _last_good_status(conn, video_id)
    for stage, (requires, _) in STAGES.items():
        if requires == status:
            return stage
    return None


def _last_good_status(conn: sqlite3.Connection, video_id: str) -> str:
    """Reconstruct progress from committed stage_runs after a failure."""
    completed = {
        row["stage"]
        for row in conn.execute(
            "SELECT DISTINCT stage FROM stage_runs WHERE video_id = ? AND status = 'ok'",
            (video_id,),
        )
    }
    status = NEW
    for stage in STAGE_ORDER:
        if stage in completed:
            status = STAGES[stage][1]
        else:
            break
    return status


def attempts_for(conn: sqlite3.Connection, video_id: str, stage: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM stage_runs WHERE video_id = ? AND stage = ? AND status = 'failed'",
        (video_id, stage),
    ).fetchone()
    return int(row["n"])


def backoff_until(attempt: int) -> str:
    """Exponential backoff: 15min, 1h, 4h."""
    minutes = 15 * (4 ** max(0, attempt - 1))
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


# --- stage bookkeeping -----------------------------------------------------


def start_stage(conn: sqlite3.Connection, video_id: str, stage: str, attempt: int) -> int:
    with transaction(conn):
        cur = conn.execute(
            "INSERT INTO stage_runs (video_id, stage, status, attempt, started_at) "
            "VALUES (?, ?, 'running', ?, ?)",
            (video_id, stage, attempt, now_iso()),
        )
    return int(cur.lastrowid)


def finish_stage_ok(
    conn: sqlite3.Connection,
    run_id: int,
    video_id: str,
    stage: str,
    duration_ms: int,
    artifact: tuple[str, Path] | None = None,
    meta: str | None = None,
) -> None:
    """Commit success: stage_run, artifact row and the status bump, atomically."""
    new_status = STAGES[stage][1]
    with transaction(conn):
        conn.execute(
            "UPDATE stage_runs SET status='ok', finished_at=?, duration_ms=?, "
            "error_class=NULL, error_text=NULL, next_retry_at=NULL WHERE id=?",
            (now_iso(), duration_ms, run_id),
        )
        if artifact is not None:
            kind, path = artifact
            digest, size = sha256_file(path)
            conn.execute(
                "INSERT INTO artifacts (video_id, kind, path, sha256, bytes, created_at, meta) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(video_id, kind) DO UPDATE SET "
                "  path=excluded.path, sha256=excluded.sha256, "
                "  bytes=excluded.bytes, created_at=excluded.created_at, "
                "  meta=excluded.meta",
                (video_id, kind, str(path), digest, size, now_iso(), meta),
            )
        conn.execute("UPDATE videos SET status=? WHERE id=?", (new_status, video_id))


def finish_stage_failed(
    conn: sqlite3.Connection,
    run_id: int,
    video_id: str,
    stage: str,
    duration_ms: int,
    error_class: str,
    error_text: str,
    max_attempts: int,
) -> str:
    """Commit failure. Returns the resulting video status."""
    attempt_count = attempts_for(conn, video_id, stage) + 1
    permanent = error_class == PERMANENT or attempt_count >= max_attempts
    status = ABANDONED if permanent else FAILED
    retry_at = None if permanent else backoff_until(attempt_count)
    with transaction(conn):
        conn.execute(
            "UPDATE stage_runs SET status='failed', finished_at=?, duration_ms=?, "
            "error_class=?, error_text=?, next_retry_at=? WHERE id=?",
            (now_iso(), duration_ms, error_class, error_text[:4000], retry_at, run_id),
        )
        conn.execute("UPDATE videos SET status=? WHERE id=?", (status, video_id))
    return status


def reap_orphan_runs(conn: sqlite3.Connection, timeout_s: int = 7200) -> int:
    """Convert abandoned `running` rows into recorded failures.

    A stage that is KILLED rather than raising — subprocess timeout, OOM kill on
    a memory-constrained box, power loss — never reaches finish_stage_failed().
    What survives is a stage_runs row stuck at 'running' and an unchanged
    videos.status. Because attempts_for() counts only 'failed' rows, the attempt
    counter never advances: max_attempts is never reached, the video never
    becomes abandoned, no backoff ever applies, and the stage re-runs at full
    API cost on every scheduled invocation, forever.

    Called at the start of every run, before the queue is claimed.
    """
    cutoff = (datetime.now(UTC) - timedelta(seconds=timeout_s)).isoformat(timespec="seconds")
    stale = list(
        conn.execute(
            "SELECT id, video_id, stage FROM stage_runs "
            "WHERE status='running' AND started_at < ?",
            (cutoff,),
        )
    )
    for row in stale:
        with transaction(conn):
            conn.execute(
                "UPDATE stage_runs SET status='failed', finished_at=?, "
                "error_class=?, error_text=? WHERE id=?",
                (now_iso(), RETRYABLE,
                 "stage process died without reporting (timeout, OOM or host restart)",
                 row["id"]),
            )
        attempt = attempts_for(conn, row["video_id"], row["stage"])
        permanent = attempt >= 3
        with transaction(conn):
            conn.execute(
                "UPDATE stage_runs SET next_retry_at=? WHERE id=?",
                (None if permanent else backoff_until(attempt), row["id"]),
            )
            conn.execute(
                "UPDATE videos SET status=? WHERE id=?",
                (ABANDONED if permanent else FAILED, row["video_id"]),
            )
    return len(stale)


def get_artifact(conn: sqlite3.Connection, video_id: str, kind: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM artifacts WHERE video_id = ? AND kind = ?", (video_id, kind)
    ).fetchone()


# --- status ----------------------------------------------------------------


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        row["status"]: row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM videos GROUP BY status ORDER BY status"
        )
    }


def oldest_pending(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT id, title, status, published_at, discovered_at FROM videos "
        "WHERE status NOT IN (?, ?) ORDER BY discovered_at ASC LIMIT 1",
        (DONE, ABANDONED),
    ).fetchone()


def abandoned_items(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT v.id, v.title, r.stage, r.error_class, r.error_text, r.finished_at
            FROM videos v
            JOIN stage_runs r ON r.video_id = v.id AND r.status = 'failed'
            WHERE v.status = ?
            GROUP BY v.id
            HAVING r.finished_at = MAX(r.finished_at)
            ORDER BY r.finished_at DESC
            LIMIT ?
            """,
            (ABANDONED, limit),
        )
    )


def last_successful_run(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT MAX(finished_at) AS t FROM stage_runs WHERE stage='publish' AND status='ok'"
    ).fetchone()
    return row["t"] if row and row["t"] else None


def stage_timings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT stage, COUNT(*) AS runs, "
            "  CAST(AVG(duration_ms) AS INTEGER) AS avg_ms, "
            "  MAX(duration_ms) AS max_ms "
            "FROM stage_runs WHERE status='ok' AND duration_ms IS NOT NULL "
            "GROUP BY stage"
        )
    )
