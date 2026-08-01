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
