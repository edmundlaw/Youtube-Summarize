"""Rolling-caption dedup is not testable by eye — hence a hand-built fixture.

The fixture reproduces the exact shape that broke two earlier implementations:
a settled cue, then a cue that redisplays it as carry-over before appending new
inline-timestamped words.
"""

from __future__ import annotations

from ytdigest.vtt import is_usable_subtitle_track, parse_cues, vtt_to_segments

ROLLING = """WEBVTT
Kind: captions
Language: yue

00:00:00.000 --> 00:00:02.000 align:start position:0%

好

00:00:02.000 --> 00:00:02.010 align:start position:0%
咦，大師咁早嘅？

00:00:02.010 --> 00:00:06.000 align:start position:0%
咦，大師咁早嘅？
每<00:00:02.200><c>個</c><00:00:02.400><c>交</c><00:00:02.600><c>易</c><00:00:02.800><c>日</c>

00:00:06.000 --> 00:00:06.010 align:start position:0%
咦，大師咁早嘅？每個交易日

00:00:06.010 --> 00:00:09.000 align:start position:0%
咦，大師咁早嘅？每個交易日
9<00:00:06.200><c>點</c><00:00:06.400><c>半</c>
"""

PLAIN = """WEBVTT

00:00:01.000 --> 00:00:03.000
Revenue grew thirteen percent.

00:00:03.000 --> 00:00:05.000
Revenue grew thirteen percent. Margins fell.
"""


def _text(segments) -> str:
    return "".join(s.text for s in segments)


def test_rolling_captions_are_not_duplicated():
    full = _text(vtt_to_segments(ROLLING))
    # Each phrase must appear exactly once, however many cues redisplayed it.
    assert full.count("咦，大師咁早嘅？") == 1
    assert full.count("每個交易日") == 1
    assert full.count("9點半") == 1


def test_rolling_captions_preserve_order_and_content():
    full = _text(vtt_to_segments(ROLLING))
    assert "每個交易日" in full
    assert full.index("每個交易日") < full.index("9點半")


def test_colloquial_cantonese_survives():
    """嘅 carries meaning; normalising it away loses information."""
    assert "嘅" in _text(vtt_to_segments(ROLLING))


def test_plain_subtitles_use_the_prefix_fallback():
    """Human subtitles have no inline timings, so the string path must run."""
    full = _text(vtt_to_segments(PLAIN))
    assert full.count("Revenue grew thirteen percent.") == 1
    assert "Margins fell." in full


def test_segments_carry_no_invented_confidence():
    """Subtitles report no confidence. Inventing one would let the validator
    treat an unverified figure as verified."""
    assert all(s.confidence is None for s in vtt_to_segments(ROLLING))


def test_timestamps_are_monotonic():
    segments = vtt_to_segments(ROLLING)
    assert segments == sorted(segments, key=lambda s: s.start)
    assert all(s.end >= s.start for s in segments)


def test_live_chat_is_not_a_transcript():
    """`live_chat` sits beside real tracks in yt-dlp output but is a chat replay."""
    assert not is_usable_subtitle_track("live_chat")
    assert is_usable_subtitle_track("yue")
    assert is_usable_subtitle_track("yue-orig")


def test_empty_and_headers_only():
    assert vtt_to_segments("") == []
    assert vtt_to_segments("WEBVTT\n\n") == []


def test_parse_cues_reads_timing():
    cues = parse_cues(ROLLING)
    assert cues
    assert cues[0].start == 0.0
    assert cues[0].end == 2.0
