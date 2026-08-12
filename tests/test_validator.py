"""Regression tests built from three failures observed in live runs.

If any of these ever go green-to-red, a real bad number is reaching the user.
"""

from __future__ import annotations

from ytdigest.interfaces import Segment
from ytdigest.normalize import build_ledger
from ytdigest.validator import (
    FAILED, PASSED, PASSED_WITH_FLAGS,
    annotate, check_text, offending, retry_instruction, verdict,
)


def _ledger(*texts: str, confidence: float | None = None):
    segs = [
        Segment(id=i, start=float(i * 10), end=float(i * 10 + 9), text=t,
                confidence=confidence)
        for i, t in enumerate(texts)
    ]
    return build_ledger(segs)


def test_fabricated_figure_is_caught():
    """The 28243 case: captions truncated, model invented an HSI target."""
    ledger = _ledger("下一個目標應該係26500度啦，26000又會錯，所以應該係八、二")
    checks = check_text("恒指目標28243", ledger)
    assert verdict(checks) == FAILED
    assert "28243" in offending(checks)


def test_figure_actually_present_passes():
    ledger = _ledger("下一個目標應該係26500度啦")
    checks = check_text("目標26500", ledger)
    assert verdict(checks) == PASSED
    assert offending(checks) == []


def test_unit_stripping_does_not_create_a_false_pass():
    """The 2000億 case: a bare '2000' elsewhere must not satisfy '2000億'."""
    ledger = _ledger("大概2000點嘅波幅", "另外講到 free cash flow 轉負")
    checks = check_text("capex達2000億", ledger)
    assert verdict(checks) == FAILED
    assert any("2000億" in f for f in offending(checks))


def test_split_number_is_flagged_not_failed():
    """The 4200|億美金 case: transcript corruption separated value from unit.
    The figure is real, so reject-outright would punish a correct summary."""
    ledger = _ledger("淨係表債務已經有4200億")          # bare magnitude, no currency
    checks = check_text("Meta表外債務4200億美金", ledger)
    assert verdict(checks) == PASSED_WITH_FLAGS
    assert offending(checks) == []
    assert any("unit differs" in c.reason for c in checks)


def test_chinese_numeral_in_source_matches_arabic_in_summary():
    """Models silently normalise 百分之十三 -> 13%. The ledger stores the
    normalised value, so the match must still succeed."""
    ledger = _ledger("free cash flow上季升咗百分之十三")
    assert verdict(check_text("FCF升13%", ledger)) == PASSED


def test_seng_normalisation_matches():
    ledger = _ledger("payout ratio維持喺四成左右")
    assert verdict(check_text("payout ratio約40%", ledger)) == PASSED


def test_low_confidence_source_is_always_flagged():
    """Even a correct figure must be marked when its source segment is shaky."""
    ledger = _ledger("升咗百分之十三", confidence=0.3)
    checks = check_text("升13%", ledger)
    assert verdict(checks) == PASSED_WITH_FLAGS


def test_clock_times_are_not_validated_as_figures():
    """9點半 is not a financial figure and must not need a ledger entry."""
    ledger = _ledger("大市今日回落")
    checks = check_text("每個交易日9點半開始", ledger)
    assert verdict(checks) == PASSED


def test_annotate_marks_only_unverified():
    ledger = _ledger("目標係26500度")
    checks = check_text("由26500升到28243", ledger)
    out = annotate("由26500升到28243", checks)
    assert "⚠︎28243" in out
    assert "⚠︎26500" not in out


def test_retry_instruction_names_the_offender_and_lists_allowed():
    ledger = _ledger("目標係26500度啦")
    checks = check_text("目標28243", ledger)
    instruction = retry_instruction(checks, ledger)
    assert "28243" in instruction
    assert "26500" in instruction
    assert "字幕於此中斷" in instruction


def test_no_numbers_is_a_pass():
    assert verdict(check_text("大市今日冇乜方向", _ledger("大市回落"))) == PASSED


def test_chinese_magnitude_in_source_matches_arabic_in_summary():
    """Regression: the real transcript said 二千億; a string-matching check
    searched for '2000億', found nothing, and wrongly reported a fabrication.
    Comparison must be on normalised value, never on text."""
    ledger = _ledger("我哋資本開始去到二千億")
    assert verdict(check_text("capex達2000億", ledger)) == PASSED


def test_colloquial_currency_flags_rather_than_fails():
    """Cantonese 蚊 is used for dollars generically. A US-listed stock quoted
    as 223蚊 is correctly rendered 223美元 — flag it, do not cry fabrication."""
    ledger = _ledger("Alp二百。223蚊啦，嗰啲就抵買啲")
    checks = check_text("Alphabet可能跌至223美元", ledger)
    assert verdict(checks) == PASSED_WITH_FLAGS
    assert offending(checks) == []


def test_lenient_json_repairs_unescaped_inner_quote():
    """Observed live: the model quoted transcript text inside a value without
    escaping, breaking the whole response. One stray quote must not cost a
    2.5-hour summary."""
    from ytdigest.summarize import loads_lenient

    broken = '{"numbers": [{"figure": "202", "context": "字幕寫"202" 唔清楚", "ts": "06:04"}]}'
    parsed = loads_lenient(broken)
    assert parsed["numbers"][0]["figure"] == "202"
    assert "202" in parsed["numbers"][0]["context"]


def test_lenient_json_leaves_valid_json_alone():
    from ytdigest.summarize import loads_lenient

    assert loads_lenient('{"a": ["x", "y"], "b": {"c": 1}}') == {"a": ["x", "y"], "b": {"c": 1}}


def test_annotation_does_not_corrupt_other_figures():
    """Regression: marking {1200億, 200億} produced ⚠︎1⚠︎200億, and {13%, 3%}
    produced ⚠︎1⚠︎3% — the safety annotation manufacturing a wrong number."""
    from ytdigest.publish import _annotate

    from ytdigest.validator import Check

    def marks(figures):
        return [Check(figure=f, unit="count", value=None,
                      verdict="missing", reason="x") for f in figures]

    assert _annotate("收入1200億，成本200億", marks(["1200億", "200億"])) == \
        "收入⚠︎1200億，成本⚠︎200億"
    assert _annotate("毛利率 13%，股息率 3%", marks(["13%", "3%"])) == \
        "毛利率 ⚠︎13%，股息率 ⚠︎3%"


def test_killed_stage_eventually_abandons():
    """Regression: a killed stage left a 'running' row, so attempts_for stayed
    0 — no backoff, no abandonment, full-cost re-run every scheduled invocation
    forever."""
    import pathlib
    import tempfile

    from ytdigest import db as D

    db_path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    conn = D.open_db(db_path, pathlib.Path("migrations"))
    conn.execute("INSERT INTO channels (id,title,added_at) VALUES ('UC1','c',?)",
                 (D.now_iso(),))
    conn.execute(
        "INSERT INTO videos (id,channel_id,title,published_at,discovered_at,status)"
        " VALUES ('v1','UC1','t','2026-01-01','2026-01-01',?)", (D.NORMALIZED,))

    for _ in range(3):
        D.start_stage(conn, "v1", "summarize",
                      D.attempts_for(conn, "v1", "summarize") + 1)
        conn.execute("UPDATE stage_runs SET started_at='2020-01-01T00:00:00+00:00'"
                     " WHERE status='running'")
        assert D.reap_orphan_runs(conn, 900) == 1

    assert D.attempts_for(conn, "v1", "summarize") == 3
    status = conn.execute("SELECT status FROM videos WHERE id='v1'").fetchone()["status"]
    assert status == D.ABANDONED
    assert D.claim_queue(conn, 10) == []


def test_hosts_are_taken_from_the_title():
    """A station trailer for another programme runs mid-episode and names that
    programme's host. The summariser picked the name up from the advert and
    attributed three claims to him at timestamps where he is never mentioned;
    the numbers verified clean, so nothing downstream caught it."""
    from ytdigest.summarize import hosts_from_title

    # Names come back canonical. parse_views compares attributions against
    # canonical_speaker(h) anyway, so resolving here changes no decision — but it
    # collapses 羅家聰 KC 博士 and a bare 羅家聰 into one entry, and the roster
    # *size* is load-bearing: at exactly one person, 「主持」 can only mean them.
    hosts = hosts_from_title(
        "CC Raga Finance：一名經人 20260723：主持：羅家聰 KC 博士、Eugene 羅尚沛、Debby 顧芷筠"
    )
    assert hosts == ["羅家聰 (KC)", "Eugene 羅尚沛", "Debby 顧芷筠"]
    assert "沈振盈" not in hosts and "沈大師" not in hosts

    # A parenthesised alias is the form actually spoken; both resolve to one person.
    assert hosts_from_title("錢錢錢打到嚟 - 主持：沈振盈(沈大師)、蔡康年") == \
        ["沈振盈 (沈大師)", "蔡康年"]
    assert hosts_from_title("no host listed") == []


def test_prompt_forbids_off_roster_attribution():
    from ytdigest.summarize import _system_prompt

    prompt = _system_prompt([], "yue", ["羅家聰 KC 博士"])
    assert "羅家聰 KC 博士" in prompt
    assert "宣傳片" in prompt          # warns about trailers naming other hosts
    assert _system_prompt([], "yue", []).count("主持") >= 1


def test_transport_errors_are_retried_in_place():
    """A dropped socket used to fail the whole stage, discarding the first
    generation that had already been paid for. It killed both videos of a
    two-video batch at the validator-retry step."""
    import httpx

    from ytdigest.config import load_config
    from ytdigest.summarize import DeepSeek

    engine = DeepSeek.__new__(DeepSeek)
    engine.cfg = load_config()
    engine.id = "stub"
    engine.base_url = "https://example.invalid"
    engine._key = "x"

    calls = {"n": 0}

    def flaky(system, user, max_tokens, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.RemoteProtocolError("peer closed connection")
        return '{"ok": true}'

    engine._once = flaky
    assert engine.complete("s", "u", 10, timeout=1) == '{"ok": true}'
    assert calls["n"] == 3


def test_long_video_timestamps_are_not_truncated():
    """Regression: _mmss took value[-5:], so on a 2.5-hour show "105:13"
    rendered as "05:13" — every citation past 100 minutes pointed 100 minutes
    too early."""
    from ytdigest.publish import _mmss

    assert _mmss("105:13") == "1:45:13"
    assert _mmss("101:34") == "1:41:34"
    assert _mmss("78:35") == "1:18:35"
    assert _mmss("42:34") == "42:34"
    assert _mmss("01:34:29") == "1:34:29"
    assert _mmss(9074) == "2:31:14"
    assert _mmss(125.0) == "02:05"


def test_ledger_sampling_covers_the_whole_video():
    """Regression: the ledger was truncated with a chronological [:120]. On a
    2.5-hour show that cut at minute 122 and hid the final 61 figures, and
    since the prompt says those are the only numbers the model may use, it was
    forbidden from discussing the densest part of the programme."""
    from ytdigest.normalize import LedgerEntry
    from ytdigest.summarize import _sample_ledger

    ledger = [
        LedgerEntry(raw_text=f"{i}億", normalized=str(i), unit="count",
                    segment_id=i, start_s=float(i * 30), confidence=None,
                    context="x")
        for i in range(500)
    ]
    sampled = _sample_ledger(ledger, 400)
    assert len(sampled) == 400
    # the tail of the video must still be represented
    assert max(e.start_s for e in sampled) > ledger[-1].start_s * 0.95
    # and when it fits, nothing is dropped
    assert len(_sample_ledger(ledger[:50], 400)) == 50


def test_views_verify_level_against_ledger():
    """A price level is rarely spoken with its unit — '跌到205' is how USD 205
    is said — so the ledger records a bare count. Demanding an exact unit match
    rejected 17 of 19 real levels. A count may back money/points, but must not
    back a percentage, where the unit carries the meaning."""
    import pathlib
    import tempfile

    from ytdigest import db as D
    from ytdigest.views import verify_level

    conn = D.open_db(pathlib.Path(tempfile.mkdtemp()) / "t.db", pathlib.Path("migrations"))
    conn.execute("INSERT INTO channels (id,title,added_at) VALUES ('UC1','c',?)", (D.now_iso(),))
    conn.execute("INSERT INTO videos (id,channel_id,title,published_at,discovered_at,status)"
                 " VALUES ('v1','UC1','t','2026-01-01','2026-01-01','done')")
    for raw, norm, unit in [("205", "205", "count"), ("13%", "13", "pct")]:
        conn.execute(
            "INSERT INTO number_ledger (video_id,raw_text,normalized,unit,segment_id,"
            "start_s,context) VALUES ('v1',?,?,?,0,0,'x')", (raw, norm, unit))

    assert verify_level(conn, "v1", 205.0, "usd")[1] is True     # bare count backs USD
    assert verify_level(conn, "v1", 205.0, "points")[1] is True  # and index points
    assert verify_level(conn, "v1", 13.0, "pct")[1] is True      # pct backs pct
    assert verify_level(conn, "v1", 205.0, "pct")[1] is False    # count must NOT back pct
    assert verify_level(conn, "v1", 999.0, "usd")[1] is False    # value absent entirely


def test_views_reject_off_roster_speakers():
    from ytdigest.views import parse_views

    payload = {"views": [
        {"speaker": "沈大師", "instrument_raw": "恒指", "direction": "long",
         "thesis": "睇好", "ts": "10:00"},
        {"speaker": "羅家聰 KC 博士", "instrument_raw": "恒指", "direction": "short",
         "thesis": "睇淡", "ts": "20:00"},
    ]}
    views = parse_views(payload, ["羅家聰 KC 博士", "Eugene 羅尚沛"])
    assert views[0].speaker is None          # 沈大師 is not on this episode
    # stored under the canonical name, not whichever spelling the title used
    assert views[1].speaker == "羅家聰 (KC)"


def test_speaker_canonicalisation():
    """The same person is introduced differently in every title and by every
    co-host. Left alone, KC's track record split across 羅家聰, KC博士 and
    羅家聰 KC 博士 and counted none of them correctly."""
    from ytdigest.views import canonical_speaker

    assert canonical_speaker("羅家聰") == "羅家聰 (KC)"
    assert canonical_speaker("KC博士") == "羅家聰 (KC)"
    assert canonical_speaker("羅家聰 KC 博士") == "羅家聰 (KC)"
    assert canonical_speaker("沈大師") == "沈振盈 (沈大師)"
    # sponsors, programme names and role words are not people
    assert canonical_speaker("哈富證券||26-07-22") is None
    assert canonical_speaker("Raga Finance") is None
    assert canonical_speaker("主持") is None
    assert canonical_speaker(None) is None


def test_conditional_calls_are_not_flattened_to_immediate():
    """KC on SK Hynix: 「如果他彈的話 我覺得應該是沽的」 — sell IF it bounces.
    Stored as a bare short, a backtest enters at the moment he spoke, which is
    exactly what he said not to do."""
    from ytdigest.views import parse_views

    views = parse_views({"views": [
        {"instrument_raw": "SK海力士", "direction": "short",
         "thesis": "仲未跌夠，如果佢彈嘅話就應該沽", "ts": "18:58"},
        {"instrument_raw": "恒指", "direction": "long",
         "thesis": "跌穿24000先入市", "ts": "05:00"},
        {"instrument_raw": "金", "direction": "long",
         "thesis": "而家就可以買入", "ts": "10:00"},
    ]}, [])
    assert views[0].entry_basis == "on_rally"
    assert views[0].stance == "bearish"
    assert views[1].entry_basis == "on_break"
    assert views[2].entry_basis == "unspecified"


def test_a_failed_retry_keeps_the_generation_already_paid_for():
    """The validator retry re-sends the whole transcript plus an instruction --
    the largest request this pipeline makes, observed dropping the connection
    on all three attempts. Raising there discards a first payload that was
    already generated, already validated, and is exactly what would be
    published had the retry run and still left offenders."""
    import pathlib

    from ytdigest.config import load_config
    from ytdigest.db import StageError
    from ytdigest.interfaces import Segment
    from ytdigest.normalize import build_ledger
    from ytdigest.summarize import PASSED_WITH_FLAGS, summarize
    import ytdigest.summarize as S

    segments = [Segment(id=0, start=0.0, end=5.0, text="恒指見二萬六",
                        lang="yue", confidence=None, flags=[])]
    ledger = build_ledger(segments)

    # First call succeeds with a figure the ledger cannot verify (forcing a
    # retry); the retry then dies on transport, as observed in the wild.
    responses = ['{"theses":[{"ts":"00:00","thesis":"恒指上望三萬九","reasoning":""}],'
                 '"actionable":[],"disagreements":[],"risks":[],"numbers":[],"views":[]}']

    class Engine:
        def __init__(self, cfg, model=None):
            pass

        def complete(self, system, user, max_tokens, timeout=None):
            if responses:
                return responses.pop(0)
            raise StageError("deepseek transport failed after 3 attempts", "retryable")

    original = S.DeepSeek
    S.DeepSeek = Engine
    try:
        cfg = load_config(pathlib.Path("."))
        payload, state, checks = summarize(cfg, segments, ledger, "yue", None, title="t")
    finally:
        S.DeepSeek = original

    assert state == PASSED_WITH_FLAGS
    assert payload["theses"][0]["thesis"] == "恒指上望三萬九"   # the first payload survived


def test_empty_completion_is_retried_in_place():
    """deepseek-v4 sometimes answers with no content at all. Observed with
    finish_reason=stop, which is not a budget problem — the same request
    succeeds next attempt. It used to fail the stage on the first occurrence."""
    from ytdigest.config import load_config
    from ytdigest.summarize import DeepSeek, _EmptyCompletion

    engine = DeepSeek.__new__(DeepSeek)
    engine.cfg = load_config()
    engine.id, engine.base_url, engine._key = "stub", "https://example.invalid", "x"

    calls = {"n": 0}

    def flaky(system, user, max_tokens, timeout):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _EmptyCompletion("empty completion (finish_reason=stop)", "retryable")
        return '{"ok": true}'

    engine._once = flaky
    assert engine.complete("s", "u", 10, timeout=1) == '{"ok": true}'
    assert calls["n"] == 2


def test_finish_reason_decides_the_remedy_not_whether_content_survived(monkeypatch):
    """A budget spent on reasoning yields either a fragment or nothing at all --
    the same failure, differing only in whether a character escaped before the
    cut. Classifying the empty case as "no content" sent it to the in-place
    retry, which re-sent the identical request three times at the identical
    budget and could never succeed."""
    import contextlib

    import httpx

    from ytdigest.summarize import _EmptyCompletion, _Truncated

    cases = [
        ("length", "", _Truncated),          # reasoning ate everything
        ("length", '{"a":', _Truncated),     # reasoning ate all but a fragment
        ("stop", "", _EmptyCompletion),      # answered with nothing; transient
    ]
    for reason, content, expected in cases:
        response = _sse(
            {"choices": [{"delta": {"content": content} if content else {},
                          "finish_reason": reason}]},
            {"choices": [], "usage": {"completion_tokens": 999,
                                      "completion_tokens_details":
                                          {"reasoning_tokens": 998}}},
        )
        monkeypatch.setattr(httpx, "stream",
                            lambda *a, **k: contextlib.nullcontext(response))
        try:
            _engine()._once("s", "u", 100, 5)
            raise AssertionError(f"expected {expected.__name__} for {reason!r}")
        except expected as exc:
            assert "reasoning_tokens=998" in str(exc), str(exc)


# --- streamed transport -----------------------------------------------------

def _engine():
    import pathlib
    from ytdigest.config import load_config
    from ytdigest.summarize import DeepSeek
    e = DeepSeek.__new__(DeepSeek)
    e.cfg = load_config(pathlib.Path("."))
    e.id, e.base_url, e._key = "stub", "https://api.example", "x"
    return e


def _sse(*frames):
    import httpx
    import json as J
    lines = [f"data: {J.dumps(f)}" for f in frames] + ["data: [DONE]"]
    return httpx.Response(200, text="\n".join(lines) + "\n",
                          request=httpx.Request("POST", "https://api.example"))


def test_stream_reassembles_content_and_carries_usage(monkeypatch):
    """usage arrives in a final frame with no choices, so finish_reason and
    usage must be accumulated separately rather than read off the last chunk."""
    import contextlib

    import httpx

    response = _sse(
        {"choices": [{"delta": {"content": '{"a"'}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": ":1}"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 9,
                                  "completion_tokens_details": {"reasoning_tokens": 7}}},
    )
    monkeypatch.setattr(httpx, "stream",
                        lambda *a, **k: contextlib.nullcontext(response))
    out = _engine()._stream({"model": "stub"}, 10)
    assert out["choices"][0]["message"]["content"] == '{"a":1}'
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 7


def test_stream_ignores_keepalive_and_unparseable_frames(monkeypatch):
    import contextlib

    import httpx

    response = httpx.Response(
        200,
        text='\n: keep-alive\n\ndata: {"choices":[{"delta":{"content":"ok"}}]}\n'
             'data: not-json\ndata: [DONE]\n',
        request=httpx.Request("POST", "https://api.example"))
    monkeypatch.setattr(httpx, "stream",
                        lambda *a, **k: contextlib.nullcontext(response))
    assert _engine()._stream({}, 10)["choices"][0]["message"]["content"] == "ok"


def test_truncated_completion_is_named_not_left_to_the_json_parser(monkeypatch):
    """A budget spent on reasoning returns a fragment. Reported as
    'unterminated string' it points nowhere near the cause -- observed as 107
    characters of a JSON object after reasoning ate 12000 tokens."""
    import contextlib

    import httpx
    import pytest as P

    from ytdigest.db import StageError

    response = _sse(
        {"choices": [{"delta": {"content": '{"theses":[{"ts":"00:0'},
                      "finish_reason": "length"}]},
        {"choices": [], "usage": {"completion_tokens": 12000,
                                  "completion_tokens_details": {"reasoning_tokens": 11978}}},
    )
    monkeypatch.setattr(httpx, "stream",
                        lambda *a, **k: contextlib.nullcontext(response))
    with P.raises(StageError, match="truncated at max_tokens"):
        _engine()._once("s", "u", 12000, 10)


def test_truncation_grows_the_budget_instead_of_repeating_the_same_request():
    """Retrying an identical request after truncation cannot help. Reasoning
    cost scales with how dense the material is, so no fixed max_tokens survives
    every video -- one 37-minute options show spent 15,795 of 16,000 tokens
    thinking and emitted 590 characters."""
    import pathlib

    from ytdigest.config import load_config
    from ytdigest.interfaces import Segment
    from ytdigest.normalize import build_ledger
    from ytdigest.summarize import _Truncated, summarize
    import ytdigest.summarize as S

    segments = [Segment(id=0, start=0.0, end=5.0, text="恒指見二萬六", lang="yue",
                        confidence=None, flags=[])]
    ledger = build_ledger(segments)
    budgets = []

    class Engine:
        def __init__(self, cfg, model=None):
            pass

        def complete(self, system, user, max_tokens, timeout=None):
            budgets.append(max_tokens)
            if len(budgets) < 2:
                raise _Truncated("completion truncated at max_tokens", "retryable")
            return ('{"theses":[],"actionable":[],"disagreements":[],'
                    '"risks":[],"numbers":[],"views":[]}')

    original = S.DeepSeek
    S.DeepSeek = Engine
    try:
        summarize(load_config(pathlib.Path(".")), segments, ledger, "yue", None, title="t")
    finally:
        S.DeepSeek = original

    assert len(budgets) == 2
    assert budgets[1] > budgets[0], "retried with the same budget"
    assert budgets[1] <= S._BUDGET_CEILING


def test_budget_growth_is_capped():
    from ytdigest.summarize import _BUDGET_CEILING, _BUDGET_GROWTH
    assert _BUDGET_GROWTH > 1.0
    assert _BUDGET_CEILING >= 20000        # room for a dense show
    assert _BUDGET_CEILING <= 100000       # but a pathological input cannot bill on


def test_a_disputed_figure_cannot_verify_clean():
    """The captions put SMIC's inflow at 29億 and our own ASR at 299億. Until a
    human resolves that, the figure must not be published as verified -- which
    is exactly what happens today, because the ledger it is checked against was
    built from the same wrong transcript."""
    from ytdigest.normalize import LedgerEntry
    from ytdigest.validator import check_text

    # unit="count": a bare "29億" with no currency word (港元/蚊/HKD/...) is
    # what find_numbers actually returns for this text -- confirmed by
    # running it directly. "hkd" would land the ledger row in the
    # unit-compatible branch instead of the exact-match branch this feature
    # lives in, and never reach the disputed check at all.
    ledger = [LedgerEntry(raw_text="29億", normalized="2900000000", unit="count",
                          segment_id=1, start_s=100.0, confidence=None, context="")]
    checks = check_text("北水淨流入29億", ledger,
                        disputed={"2900000000": "29900000000"})

    assert checks and checks[0].verdict == "flagged"
    assert "29900000000" in checks[0].reason


def test_an_agreed_figure_still_passes():
    from ytdigest.normalize import LedgerEntry
    from ytdigest.validator import check_text

    ledger = [LedgerEntry(raw_text="56億", normalized="5600000000", unit="count",
                          segment_id=1, start_s=100.0, confidence=None, context="")]
    checks = check_text("淨流入56億", ledger, disputed={})
    assert checks and checks[0].verdict == "ok"
