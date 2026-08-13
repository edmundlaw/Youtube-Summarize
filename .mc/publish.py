#!/usr/bin/env python3
"""Publish what ytdigest has produced, for the Mission Control panel.

Writes <repo>/.mc/output.json following Mission Control's contract. The panel
reads that file on its own schedule; it never runs this script and never
reaches into this project's database.

THE RULES THIS SCRIPT FOLLOWS.

1.  IT STARTS NOTHING. No import from `ytdigest`, no network call, no model
    load. Importing the package would be enough to matter here: the pipeline
    runs one subprocess per stage precisely because MLX does not reliably hand
    unified memory back, and this machine has 16 GB shared with a dozen other
    services. It also holds a flock for single-instance; nothing here goes
    near it.

2.  IT OPENS THE DATABASE READ-ONLY. sqlite3 in mode=ro, so it cannot write,
    migrate, or take a lock from the pipeline that owns the file. The database
    is in WAL mode and a scheduled run may be mid-write.

3.  IT CANNOT FAIL THE PIPELINE. The caller discards its exit code. If it
    cannot produce a file it says so on stderr and stops, so the panel shows
    ytdigest as not publishing -- which is true -- rather than showing a stale
    file as current.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "state.db"
OUT_PATH = ROOT / ".mc" / "output.json"

# launchd runs the pipeline at 22:30 and 06:30 HKT, so the longest legitimate
# gap between republishes is the 16 hours from the morning run to the evening
# one. The panel calls the file stale at three times this.
REFRESH_S = 16 * 3600

# A list Edmund reads, not an export.
LIMIT = 30


def _read_payload(path_text: str) -> dict | None:
    """The summary artifact, or None. A missing file is not an error worth
    failing over -- artifacts can be pruned while the database row remains."""
    path = pathlib.Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rows() -> list[dict]:
    """Recently published summaries, newest first, grouped by channel."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            "SELECT v.id, v.title, v.published_at, c.title AS channel, a.path,"
            "       (SELECT COUNT(*) FROM number_ledger n"
            "         WHERE n.video_id = v.id AND n.crosscheck = 'disputed')"
            "         AS disputed"
            "  FROM videos v"
            "  JOIN channels c ON c.id = v.channel_id"
            "  JOIN artifacts a ON a.video_id = v.id AND a.kind = 'summary'"
            " WHERE v.status = 'done'"
            " ORDER BY v.published_at DESC"
            " LIMIT ?", (LIMIT,)).fetchall()
    finally:
        conn.close()

    items = []
    for r in found:
        item = {
            "title": r["title"],
            "group": r["channel"] or None,
            "url": f"https://youtu.be/{r['id']}",
            "at": (r["published_at"] or "") or None,
        }

        # The summary line is the episode's leading thesis, verbatim. If the
        # model produced none, the field is left out rather than filled with a
        # stand-in -- an episode with no thesis and an episode whose thesis we
        # invented look identical once published, and only one of them is true.
        payload = _read_payload(r["path"]) or {}
        theses = (payload.get("payload") or {}).get("theses") or []
        summary = None
        if theses and isinstance(theses[0], dict):
            summary = (theses[0].get("thesis") or "").strip() or None

        # Figures where our own ASR heard something different from the
        # captions. Worth seeing here: it is the one thing in a digest that
        # asks Edmund to go back to the source and listen.
        if r["disputed"]:
            mark = f"⚠︎ {r['disputed']} 個數字有爭議"
            summary = f"{mark} · {summary}" if summary else mark
        if summary:
            item["summary"] = summary

        items.append({k: v for k, v in item.items() if v is not None})
    return items


def main() -> int:
    if not DB_PATH.exists():
        print(f"[mc-publish] no database at {DB_PATH}", file=sys.stderr)
        return 1
    try:
        items = rows()
    except sqlite3.Error as exc:
        print(f"[mc-publish] could not read the database: {exc}", file=sys.stderr)
        return 1

    document = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": "reading",
        "title": "Summaries published",
        "refresh_s": REFRESH_S,
        # An empty list is a real answer: "produced nothing" and "not
        # publishing" are different facts and the panel shows them differently.
        "items": items,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Written atomically: the panel reads on its own schedule and must never
    # catch a half-written file.
    tmp = OUT_PATH.with_name(".output.json.tmp")
    tmp.write_text(json.dumps(document, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, OUT_PATH)
    print(f"[mc-publish] {len(items)} summaries -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
