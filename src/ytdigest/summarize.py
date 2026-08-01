"""Summarisation, with the ledger supplied as the sole source of figures.

Two things here earn their complexity:

* map-reduce, so a 2.5-hour transcript does not have to fit one context. The
  ledger is passed to BOTH levels, so the reduce step never has to recall a
  number from a map-level summary.
* the validator retry loop. The prompt already forbids inventing figures and
  the model does it anyway when captions truncate mid-sentence, so the
  regeneration pass names the offending figures explicitly.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from .config import Config, load_glossary
from .db import RETRYABLE, StageError
from .normalize import LedgerEntry
from .validator import (
    PASSED_WITH_FLAGS, check_text, offending, retry_instruction, verdict,
)

#: v3 added the attribution, reported-speech and hedge rules. Bumping this
#: matters for the track record: views carrying an older version were attributed
#: by guesswork and must not be pooled with v3 ones when judging a speaker.
PROMPT_VERSION = "v3"


class _EmptyCompletion(StageError):
    """The model answered, but with no content.

    Its own class so `complete()` can retry it in place. It is a StageError
    subclass so that callers which do not care still handle it as one.
    """


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


class DeepSeek:
    """Summarizer implementation for DeepSeek's OpenAI-compatible endpoint.

    Note: these models spend reasoning tokens before emitting output, and those
    count against max_tokens. A budget sized only for the answer returns an
    empty string with finish_reason 'length' — silently, and it looks like a
    parse bug rather than a budget one.
    """

    def __init__(self, cfg: Config, model: str | None = None):
        self.cfg = cfg
        self.id = model or cfg.get("summarize", "model", "deepseek-v4-flash")
        self.base_url = cfg.get("summarize", "base_url", "https://api.deepseek.com")
        self._key = cfg.require_secret("DEEPSEEK_API_KEY")

    def complete(self, system: str, user: str, max_tokens: int = 10000,
                 timeout: float = 300.0, attempts: int = 3) -> str:
        """Call the model, retrying transport-level failures in place.

        DeepSeek drops the connection mid-response often enough to matter
        ("peer closed connection without sending complete message body").
        Observed on three separate runs, and it killed both videos of a
        two-video batch at the validator-retry step. Letting it propagate
        fails the whole stage and discards the first generation, which has
        already been paid for — so a transient socket error costs a full
        re-run of everything.
        """
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._once(system, user, max_tokens, timeout)
            except (httpx.RemoteProtocolError, httpx.ReadError,
                    httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError) as exc:
                last = exc
                if attempt < attempts:
                    time.sleep(2 ** attempt)
            except _EmptyCompletion as exc:
                # A completion with no content is as non-deterministic as the
                # malformed JSON already retried a layer up: the same request
                # succeeds on the next attempt. Letting it through failed the
                # stage on the first try and discarded everything.
                last = exc
                if attempt < attempts:
                    time.sleep(2 ** attempt)
        raise StageError(
            f"deepseek call failed after {attempts} attempts: {last}", RETRYABLE
        )

    def _once(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self.id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # 429 and 5xx are worth retrying; a 400 means our request is wrong.
            klass = RETRYABLE if status == 429 or status >= 500 else "permanent"
            raise StageError(f"deepseek {status}: {exc.response.text[:300]}", klass) from exc
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                httpx.ReadTimeout, httpx.WriteError):
            # Must reach complete()'s retry loop, not be wrapped here.
            raise
        except Exception as exc:
            raise StageError(f"deepseek request failed: {exc}", RETRYABLE) from exc

        choice = payload["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        if not content:
            # Report what actually happened rather than always blaming the
            # budget. deepseek-v4 returns reasoning in a separate field and
            # counts it against max_tokens, so an empty answer has two quite
            # different causes and only one of them is fixed by raising it:
            #   finish_reason=length -> reasoning consumed the budget
            #   finish_reason=stop   -> the model simply returned nothing,
            #                           which is transient and worth retrying
            usage = payload.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            reason = choice.get("finish_reason")
            advice = ("raise summarize.max_tokens — reasoning tokens count "
                      "against it" if reason == "length"
                      else "model returned no content; transient")
            raise _EmptyCompletion(
                f"empty completion (finish_reason={reason}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"reasoning_tokens={details.get('reasoning_tokens')}, "
                f"max_tokens={max_tokens}); {advice}",
                RETRYABLE,
            )
        return content


def hosts_from_title(title: str) -> list[str]:
    """Names listed after 主持 / 主持人 in the video title.

    Needed because a station trailer for another programme runs mid-episode and
    names that programme's host. On a real 55-minute episode the summariser
    picked the name up from the advert and attributed three claims to him at
    timestamps where he is never mentioned — the numbers verified clean, so
    nothing downstream could catch it. Wrong attribution is fatal to a
    prediction track record: it credits calls to someone who never made them.
    """
    # 主持 (host) and 嘉賓 (guest) both identify who is speaking. Channels are
    # inconsistent: RagaFinance uses 主持：, Kinnis uses 嘉賓：, and 1號月台 uses
    # neither — it tags the guest with a #hashtag instead.
    match = re.search(r"(?:主持人?|嘉賓)\s*[:：]\s*(.+)$", title)
    if not match:
        tags = re.findall(r"#([^\s#]+)", title)
        known = [t for t in tags if not t.lower().startswith(("kctalk", "kc博士"))]
        return known[:3]
    tail = re.split(r"[|｜\[]", match.group(1))[0]
    names: list[str] = []
    for part in re.split(r"[、,，/／&]|\s+同\s+", tail):
        part = part.strip()
        if not part:
            continue
        # 沈振盈(沈大師): the parenthesised alias is the name actually spoken,
        # so both forms belong in the roster.
        alias = re.search(r"[(（]([^)）]+)[)）]", part)
        base = re.sub(r"[(（][^)）]*[)）]", "", part).strip()
        if base:
            names.append(base)
        if alias and alias.group(1).strip():
            names.append(alias.group(1).strip())
    return names


def _system_prompt(ledger: list[LedgerEntry], lang: str,
                   hosts: list[str] | None = None,
                   voice_identified: bool = False) -> str:
    target = "English" if lang == "en" else "繁體中文（香港）"
    glossary = ", ".join(load_glossary())
    allowed = "\n".join(
        f"  - {e.raw_text}  [{_mmss(e.start_s)}]" for e in _sample_ledger(ledger, 400)
    )
    span = max((e.start_s for e in ledger), default=0.0)
    duration = _mmss(span) if span else "未知"
    host_rule = (
        "呢一集嘅主持只有：" + "、".join(hosts) + "。"
        if hosts else "片名冇列出主持，所以一律寫「主持」，唔好用任何人名。"
    )
    # With voice identification the name on each line is measured, not guessed,
    # so the model must copy it rather than reason about it. Without it, the
    # model has no way to know who spoke and must not pretend otherwise.
    voice_rule = (
        "每一行字幕前面嘅名，係用聲紋認出嚟嘅，唔係估。\n"
        "  ‧ 寫住人名嗰行 → 就係嗰個人講，直接用返個名。\n"
        "  ‧ 寫住「主持」嗰行 → 認唔出（可能幾個人一齊講，或者未錄過佢聲）。\n"
        "    呢啲行一律寫「主持」，唔准按上文下理估係邊個。\n"
        "唔准改行頭嗰個名，亦唔准將「主持」嗰行寫成某個人講。"
        if voice_identified else
        "字幕係冇標明邊個講嘢嘅。所以【預設一律寫「主持」】。\n"
        "只有以下情況先可以寫人名：\n"
        "  ‧ 有人叫佢個名（例：「KC你點睇？」→ 之後嗰段係 KC 講）\n"
        "  ‧ 佢自報身份（例：「我上個禮拜喺韓國……」而片名講明只有佢去過）\n"
        "冇上面嘅憑據就寫「主持」。寫錯邊個講，比冇寫名嚴重得多。"
    )
    return f"""你為香港投資者提煉財經影片。輸出語言：{target}。

以下詞彙保持英文原樣，唔准翻譯：{glossary}
股票代號（0700.HK、BABA）照原文。
保留主持嘅粵語口語（嘅／咗／喺／唔／哋），唔好改成書面語。

【最重要：唔好寫「內容摘要」，要寫「主持實際主張咗乜、點解」】
禁止呢類句子（適用於任何一集＝等於冇講嘢）：
  ✗「討論AI泡沫」✗「分析港股走勢」✗「提醒投資者注意風險」
測試：一句話如果可以原封不動放喺任何一日任何一集，刪咗佢。
要寫：邊個講、具體主張、佢嘅理由。主持之間有分歧要特別指出。

【覆蓋鐵律】
呢條片長 {duration}。你要覆蓋成條片，由頭到尾。
唔好淨係講頭三十分鐘就算數 —— 好多節目最重要嘅分析喺後半段。
每一個大段落（大約每二十分鐘）至少要有一個論點或者可操作內容。

【講者鐵律】
{host_rule}
唔准將主張歸咎於名單以外嘅人。節目中間會播其他節目嘅宣傳片，
入面會提到第二個節目嘅主持名。嗰啲名唔屬於呢一集，唔可以當佢哋喺度講嘢。

{voice_rule}

【引述唔等於主張】
主持之間會互相引用、質問對方之前講過嘅嘢。例如
「你唔係話120蚊咩，我記得」——講呢句嗰個人係喺度*問返*對方，
唔係佢自己睇120蚊。呢種情況：唔可以當作講嘢嗰個人嘅觀點，
亦唔可以當作被問嗰個人而家嘅觀點（佢可能已經改咗主意）。
除非佢自己親口再確認，否則唔好當係任何人嘅 view。

【保留佢嘅保留】
「未必」「唔一定」「可能」「睇怕」係佢刻意留低嘅餘地，要照寫。
講「係咪成個下半年差就未必」＝佢*唔認為*成個下半年一定差，
唔可以寫成「下半年可能回落」。
講「彈返啲但係唔夠力」＝彈but弱，唔可以只寫「有反彈」。
將佢嘅對沖講法拉直做預測，係最常見亦最誤導嘅錯。

【數字鐵律】
你只可以使用以下喺字幕入面真正出現過嘅數字：
{allowed}

唔准推算、唔准四捨五入、唔准補完。
如果某段字幕斷咗、殘缺、講到一半冇咗，就唔好寫個數字，
改為寫「字幕於此中斷」。寧願少寫，都唔可以砌一個數字出嚟。

JSON 字串值裏面唔好用半形雙引號 \" ，要引用字幕原文就用「」。

另外要抽取「市場觀點」(views)：每一個具體嘅睇好/睇淡/中性判斷，一個標的一行。
  instrument_raw : 主持點叫嗰個標的（原文，例如 恒指、騰訊、日圓、金）
  direction      : long / short / neutral / avoid / exit
  level_value    : 只可以填字幕入面真正出現過嘅數字，冇就留空
  level_type     : target / support / resistance / entry / stop / valuation
  horizon        : intraday / days / weeks / months / quarters / year
  entry_basis    : immediate（而家就做）/ on_rally（等彈先沽）/ on_dip（等跌先買）/
                   on_break（穿位先做）/ on_confirmation（等確認）/ unspecified
  condition      : 如果係有條件先做，用主持原話寫低個條件，例如「如果佢彈嘅話」
  stance         : bullish / bearish / neutral（佢對呢個標的嘅基本睇法）
【重要】主持話「等反彈先沽」同「而家沽」係兩件事。
唔好將有條件嘅判斷寫成即時判斷 —— 要照佢講嘅條件記低。
  speaker        : 名單入面嘅主持，或者「主持」（唔肯定邊個講就用呢個）。
                   照【講者鐵律】——冇憑據就寫「主持」，唔准估。
純粹講宏觀背景、冇具體判斷嘅，唔好當 view。
主持引述／質問對方之前講過嘅數字，唔算 view（見【引述唔等於主張】）。

輸出 JSON：
{{"views":[{{"ts":"MM:SS","speaker":"","instrument_raw":"","direction":"",
  "conviction":"high|medium|low","thesis":"","reasoning":"",
  "level_type":"","level_value":null,"level_unit":"hkd|usd|pct|points","horizon":"",
  "entry_basis":"","condition":"","stance":""}}],
 "actionable":[{{"ts":"MM:SS","ticker":"如有","claim":""}}],
 "theses":[{{"ts":"MM:SS","thesis":"","reasoning":""}}],
 "disagreements":[{{"ts":"MM:SS","detail":"寫邊個持邊個立場前，一樣要跟【講者鐵律】"}}],
 "risks":[{{"ts":"MM:SS","risk":""}}],
 "numbers":[{{"figure":"","context":"","ts":"MM:SS"}}]}}"""


def loads_lenient(text: str) -> dict:
    """Parse model JSON, repairing the one break it reliably produces.

    Observed in a live run: the model quoted transcript text inside a value
    without escaping, e.g.

        "context": "字幕寫"202" 唔清楚。",

    which is invalid JSON from that character onward. `response_format:
    json_object` does not prevent it. Rather than fail a whole 2.5-hour
    summary on one stray quote, walk the text and escape interior quotes —
    a quote is closing only if the next non-space character is one of , : } ]
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    repaired: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            repaired.append(char)
            escaped = False
            continue
        if char == "\\":
            repaired.append(char)
            escaped = True
            continue
        if char == '"':
            if not in_string:
                in_string = True
                repaired.append(char)
                continue
            following = text[index + 1 :]
            stripped = following.lstrip(" \t\r\n")
            if stripped[:1] in {",", ":", "}", "]", ""}:
                in_string = False
                repaired.append(char)
            else:
                repaired.append('\\"')  # interior quote — escape it
            continue
        repaired.append(char)

    candidate = "".join(repaired)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # A literal newline inside a string is the other break the model produces.
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError as exc:
        raise StageError(
            f"model returned unparseable JSON even after repair: {exc}", RETRYABLE
        ) from exc


def _dedupe(ledger: list[LedgerEntry]) -> list[LedgerEntry]:
    seen: set[tuple[str, str]] = set()
    out: list[LedgerEntry] = []
    for entry in ledger:
        key = (entry.raw_text, entry.unit)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def _sample_ledger(ledger: list[LedgerEntry], limit: int) -> list[LedgerEntry]:
    """Choose which figures to show the model, covering the whole video.

    This used to be a plain [:120] on a chronologically ordered ledger. On a
    2.5-hour show that cut at minute 122 and hid the final 61 figures — and
    since the prompt tells the model those are the ONLY numbers it may use, it
    was structurally forbidden from discussing the last half hour. That half
    hour was the densest part of the programme (71 figures in one 15-minute
    bucket against 9 in the opening), so the digest silently omitted the
    substance of the episode.

    When the ledger fits, show all of it. When it does not, sample evenly
    across the timeline so every part of the video stays reachable.
    """
    entries = _dedupe(ledger)
    if len(entries) <= limit:
        return entries
    stride = len(entries) / limit
    return [entries[int(i * stride)] for i in range(limit)]


def _transcript_text(segments, speakers: dict[float, str] | None = None) -> str:
    """Render for the prompt, prefixing the speaker where voice identified one.

    Only segments that cleared the voice threshold get a name. Everything else
    is rendered `[主持]`, which is a real answer rather than a gap: it tells the
    model that this line's speaker is genuinely unknown, so it must not invent
    one. Before this existed the model saw an unlabelled wall of text on a
    three-host show and attributed by guesswork.
    """
    lines = []
    for s in segments:
        who = (speakers or {}).get(round(float(s.start), 3))
        lines.append(f"[{_mmss(s.start)}] {who or '主持'}：{s.text}"
                     if speakers is not None else f"[{_mmss(s.start)}] {s.text}")
    return "\n".join(lines)


def speaker_map(conn, video_id: str) -> dict[float, str] | None:
    """Voice-identified speaker per segment start, or None if never run.

    None and an all-unattributed map mean different things: the first says
    identification has not happened, the second says it happened and refused.
    """
    rows = list(conn.execute(
        "SELECT start_s, speaker FROM segment_speakers WHERE video_id = ?",
        (video_id,)))
    if not rows:
        return None
    return {round(float(r["start_s"]), 3): r["speaker"] for r in rows if r["speaker"]}


def estimate_tokens(text: str) -> int:
    """Rough token count for a mixed CJK/Latin transcript.

    CJK is close to one token per character; Latin runs about four characters
    per token. The map-reduce trigger previously compared a *character* count
    against a threshold named in tokens and then doubled it, so it fired at
    24,000 characters — a quarter of the intended size for English, and well
    below a routine 2.5-hour Cantonese show.
    """
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk + max(0, len(text) - cjk) // 4


def _flatten(payload: dict) -> str:
    """Every free-text field the validator must inspect."""
    parts: list[str] = []
    for item in payload.get("actionable", []):
        parts.append(str(item.get("claim", "")))
    for item in payload.get("theses", []):
        parts += [str(item.get("thesis", "")), str(item.get("reasoning", ""))]
    for item in payload.get("disagreements", []):
        parts.append(str(item.get("detail", "")))
    for item in payload.get("risks", []):
        parts.append(str(item.get("risk", "")))
    for item in payload.get("numbers", []):
        parts += [str(item.get("figure", "")), str(item.get("context", ""))]
    for item in payload.get("views", []):
        parts += [str(item.get("thesis", "")), str(item.get("reasoning", ""))]
    # A bare newline lets a pattern span two unrelated fields: a figure
    # ending one field joined the 股 opening the next, yielding phantom
    # "165\n股" entries that can never verify. A non-space separator
    # blocks the match.
    return "\n。\n".join(p for p in parts if p.strip())


def summarize(
    cfg: Config,
    segments,
    ledger: list[LedgerEntry],
    lang: str,
    log=None,
    title: str = "",
    speakers: dict[float, str] | None = None,
) -> tuple[dict, str, list]:
    """Generate, validate, retry once, and return (payload, state, checks)."""
    engine = DeepSeek(cfg)
    system = _system_prompt(ledger, lang, hosts_from_title(title),
                            voice_identified=speakers is not None)
    body = _transcript_text(segments, speakers)
    max_tokens = int(cfg.get("summarize", "max_tokens", 10000))

    threshold = int(cfg.get("summarize", "map_reduce_threshold_tokens", 40000))
    if estimate_tokens(body) > threshold:
        body = _map_reduce(engine, system, segments, max_tokens, cfg, log,
                           speakers)

    # Malformed JSON from the model is non-deterministic — the same input
    # parses fine on the next attempt. Regenerating immediately is far cheaper
    # than failing the stage and waiting for the next scheduled run.
    for attempt in range(1, 4):
        try:
            payload = loads_lenient(engine.complete(system, body, max_tokens))
            break
        except StageError:
            if attempt == 3:
                raise
            if log:
                log.warning("summarize.unparseable_json", attempt=attempt)
    checks = check_text(_flatten(payload), ledger)
    state = verdict(checks)

    if offending(checks):
        if log:
            log.warning("summarize.retry", offenders=offending(checks))
        instruction = retry_instruction(checks, ledger)
        # Retry on the SAME model. Escalating to deepseek-v4-pro was tried and
        # dropped: on a full-transcript retry it dropped the connection on all
        # three attempts for both sample videos, and it translates the
        # protected glossary terms that flash preserves — so it was both more
        # fragile and lower quality on the one axis that matters here.
        retry_engine = engine
        # A failed retry must not cost us the generation we already have. The
        # retry sends the whole transcript again plus the instruction, and that
        # request is the largest this pipeline makes -- it has been observed
        # dropping the connection on all three attempts. Losing the first
        # payload there means paying for it, discarding it, and publishing
        # nothing, when the first payload is exactly what we would publish if
        # the retry ran and still left offenders.
        try:
            retried = loads_lenient(
                retry_engine.complete(system, f"{body}\n\n---\n{instruction}",
                                      max_tokens)
            )
        except StageError as exc:
            if log:
                log.warning("summarize.retry_failed", error=str(exc)[:200])
            return payload, PASSED_WITH_FLAGS, checks

        payload = retried
        checks = check_text(_flatten(payload), ledger)
        state = verdict(checks)
        # Still unverifiable after one retry: publish with the figures marked
        # rather than dropping the summary. A flagged figure is useful; a
        # silently-published wrong one is not.
        if offending(checks):
            state = PASSED_WITH_FLAGS
            if log:
                log.warning("summarize.still_unverified", offenders=offending(checks))

    return payload, state, checks


def _map_reduce(engine: DeepSeek, system: str, segments, max_tokens: int,
                cfg: Config, log=None,
                speakers: dict[float, str] | None = None) -> str:
    """Chunk-level summaries, then a synthesis pass.

    The ledger travels in `system`, so it is present at both levels and the
    reduce step never recalls a number from a map-level summary.
    """
    size = int(cfg.get("summarize", "chunk_tokens", 20000))
    chunks: list[list] = []
    current: list = []
    length = 0
    for seg in segments:
        current.append(seg)
        length += estimate_tokens(seg.text)
        if length >= size:
            chunks.append(current)
            current, length = [], 0
    if current:
        chunks.append(current)

    if log:
        log.info("summarize.map_reduce", chunks=len(chunks))

    partials: list[str] = []
    for index, chunk in enumerate(chunks):
        try:
            text = engine.complete(
                system + "\n\n（呢個係影片其中一段，先做分段摘要）",
                _transcript_text(chunk, speakers),
                max_tokens,
            )
        except StageError as exc:
            # One failed chunk must not discard the ones already paid for.
            if log:
                log.warning("summarize.chunk_failed", chunk=index + 1,
                            of=len(chunks), error=str(exc)[:200])
            if not partials:
                raise
            partials.append(f"--- 第{index + 1}段 ---\n（呢一段摘要失敗，內容從缺）")
            continue
        partials.append(f"--- 第{index + 1}段 ---\n{text}")
    return "\n\n".join(partials)
