"""The digest reports what needs checking, never what passed.

A "✓ verified" tick on every figure trains the eye to skim past the one that
carries a warning, and `validator: passed` on a summary the validator only
ever half-checked reads as a guarantee the pipeline cannot make. Both are
removed here; the markdown file keeps the full audit trail.
"""

from ytdigest.publish import render_telegram
from ytdigest.validator import Check


VIDEO = {"id": "abc123", "title": "測試", "channel_title": "Test", "published_at": "2026-08-09"}


def _ok(figure):
    return Check(figure=figure, unit="", value=None, verdict="ok", reason="")


def _missing(figure):
    return Check(figure=figure, unit="", value=None, verdict="missing", reason="")


def _payload(*figures):
    return {"numbers": [{"figure": f, "context": f"提到{f}"} for f in figures]}


def test_no_success_confirmations():
    text = render_telegram(VIDEO, _payload("13%", "40%"), [_ok("13%"), _ok("40%")])
    assert "validator" not in text
    assert "✓" not in text
    assert "核對到" not in text
    assert "13%" in text and "40%" in text


def test_legend_absent_when_nothing_is_flagged():
    text = render_telegram(VIDEO, _payload("13%"), [_ok("13%")])
    assert "需核實" not in text


def test_unverified_figure_still_carries_its_warning():
    text = render_telegram(VIDEO, _payload("13%", "999%"), [_ok("13%"), _missing("999%")])
    assert "⚠︎ <b>999%</b>" in text
    assert "⚠︎ <b>13%</b>" not in text
    assert "需核實" in text           # legend earns its line
    assert "有數字核對唔到" in text   # and the summary line above still fires


def test_figure_with_no_check_is_treated_as_unverified():
    """Silence from the validator is not a pass. A figure it never reached
    must not render the same as one it confirmed."""
    text = render_telegram(VIDEO, _payload("13%"), [])
    assert "⚠︎ <b>13%</b>" in text
