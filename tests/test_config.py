"""The shipped config.toml must actually contain what the code reads.

Every setting has a code-side default, which is good for robustness and bad
for detectability: a key that goes missing does not fail, it silently reverts.
That happened -- inserting a `[voice]` section dropped the `[summarize]`
header, so every summarize key became a voice key and `max_tokens` fell back
from 12000 to the code default of 10000. Nothing raised. The only symptom was
a truncated completion much later, and only under a prompt that had grown.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest

CONFIG = pathlib.Path("config/config.toml")

#: Settings the code reads with a fallback. Present here means the shipped file
#: is authoritative; absent means the fallback is silently in charge.
REQUIRED = {
    "paths": ["data_dir"],
    "summarize": ["provider", "model", "escalate_model", "base_url",
                  "max_tokens", "map_reduce_threshold_tokens", "chunk_tokens"],
    "voice": ["enabled", "threshold", "margin", "keep_audio"],
    "asr": ["prefer_manual_subs", "keep_audio"],
    "publish": ["telegram"],
}


@pytest.fixture(scope="module")
def config():
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("section,keys", sorted(REQUIRED.items()))
def test_section_has_its_keys(config, section, keys):
    assert section in config, f"[{section}] missing from {CONFIG}"
    missing = [k for k in keys if k not in config[section]]
    assert not missing, (
        f"[{section}] is missing {missing}. A dropped section header silently "
        "reassigns its keys to the section above, and the code falls back to "
        "its own defaults without complaining."
    )


def test_no_section_has_absorbed_another(config):
    """A missing header does not produce a parse error -- the orphaned keys
    just join the previous table. The signature is one section carrying keys
    that belong to another."""
    owners = {key: section
              for section, keys in REQUIRED.items() for key in keys
              if key not in {"keep_audio"}}          # legitimately in two places
    for section, table in config.items():
        for key in table:
            expected = owners.get(key)
            if expected and expected != section:
                pytest.fail(
                    f"[{section}] contains '{key}', which belongs to "
                    f"[{expected}] — the [{expected}] header was probably lost.")


def test_voice_threshold_matches_the_calibrated_default(config):
    """config.toml and voice.py must not drift apart: the calibration that
    justifies this number lives in voice.py, and a config that quietly
    disagrees would attribute at an uncalibrated threshold."""
    from ytdigest import voice
    assert config["voice"]["threshold"] == pytest.approx(voice.DEFAULT_THRESHOLD)
    assert config["voice"]["margin"] == pytest.approx(voice.DEFAULT_MARGIN)


# --- backfill listing -------------------------------------------------------

def test_upload_listing_never_invents_a_publish_date():
    """A flat upload listing carries no date -- only id and title, every other
    field null. Defaulting to "now" was tried and is actively dangerous: every
    view's stated_at derives from published_at, so a backfilled opinion would be
    dated to its import day and graded against the wrong price window."""
    from ytdigest.sources import youtube

    class FakeYDL:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {"entries": [
                {"id": "abc", "title": "A show", "upload_date": None,
                 "duration": None, "timestamp": None},
                {"id": "def", "title": "Another"},
                {"title": "no id at all"},
            ]}

    original = youtube._ydl
    youtube._ydl = lambda **kw: FakeYDL()
    try:
        refs = youtube.list_uploads("UC123")
    finally:
        youtube._ydl = original

    assert [r.id for r in refs] == ["abc", "def"]        # entry without id dropped
    assert all(r.published_at is None for r in refs)
    assert [r.title for r in refs] == ["A show", "Another"]


# --- backfill must not flood the channel ------------------------------------

def test_old_videos_do_not_trigger_a_notification():
    """Backfilling a back catalogue queues hundreds of videos through the same
    pipeline as today's upload. Without this, each one arrives as its own
    Telegram message -- 320 of them for the queue measured on this corpus."""
    import pathlib
    from datetime import UTC, datetime, timedelta

    from ytdigest.config import load_config
    from ytdigest.runner import _is_newsworthy

    cfg = load_config(pathlib.Path("."))
    now = datetime.now(UTC)
    fresh = {"published_at": (now - timedelta(hours=6)).isoformat()}
    old = {"published_at": (now - timedelta(days=200)).isoformat()}
    assert _is_newsworthy(cfg, fresh) is True
    assert _is_newsworthy(cfg, old) is False


def test_unknown_or_unparseable_age_is_treated_as_current():
    """Silence is the wrong default for an unknown date: a real new upload must
    never be dropped because its timestamp could not be read."""
    import pathlib

    from ytdigest.config import load_config
    from ytdigest.runner import _is_newsworthy

    cfg = load_config(pathlib.Path("."))
    assert _is_newsworthy(cfg, {"published_at": None}) is True
    assert _is_newsworthy(cfg, {"published_at": "not a date"}) is True
    assert _is_newsworthy(cfg, {}) is True


def test_notify_window_is_configured():
    import pathlib
    import tomllib
    d = tomllib.loads(pathlib.Path("config/config.toml").read_text())
    assert "notify_within_days" in d["publish"]


def test_ffmpeg_is_located_without_relying_on_PATH():
    """launchd runs with a minimal PATH that excludes Homebrew, so relying on
    PATH means audio extraction works by hand and fails under the scheduler.
    It did: every scheduled run logged "ffprobe and ffmpeg not found" and
    silently produced no speaker attribution."""
    import shutil

    from ytdigest.sources import youtube

    original = shutil.which
    shutil.which = lambda name: None          # simulate the launchd PATH
    try:
        location = youtube.ffmpeg_dir()
    finally:
        shutil.which = original
    assert location in youtube._FFMPEG_DIRS or location is None
    # On this machine ffmpeg is installed, so the fallback must find it.
    import pathlib
    if any((pathlib.Path(d) / "ffmpeg").exists() for d in youtube._FFMPEG_DIRS):
        assert location is not None, "installed ffmpeg not found without PATH"


def test_ydl_passes_ffmpeg_location_when_known():
    from ytdigest.sources import youtube
    if youtube.ffmpeg_dir() is None:
        return
    ydl = youtube._ydl()
    assert ydl.params.get("ffmpeg_location") == youtube.ffmpeg_dir()
