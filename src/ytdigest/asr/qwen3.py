"""Qwen3-ASR via mlx-audio.

Chosen over Whisper by measurement, not preference. On two windows of a real
episode Whisper scored zero on 口語 in every single run -- it renders 嘅咗喺唔
into Mandarin-phrased Simplified Chinese, which AGENTS.md forbids because it
changes meaning. It also turned 資金流 into 紫金流 purely because 紫金礦業 was in
the bias list, and one run degenerated into several hundred repetitions of one
character. Qwen keeps the Cantonese, gets 中芯國際 and 淨流入 right where the
captions never have, and exposes `repetition_penalty`.

Full details: docs/superpowers/specs/2026-08-11-asr-bake-off-results.md
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..interfaces import Segment, SpeechRegion
from ..logging import get_logger

MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-bf16"
LANG = "yue"
#: Qwen degenerates into repetition without this. Whisper had no equivalent and
#: produced several hundred repetitions of one character on a prompted run.
REPETITION_PENALTY = 1.1

log = get_logger()


class Qwen3ASRMLX:
    id = "qwen3-asr-1.7b-bf16"

    def __init__(self, cfg):
        self._model_id = cfg.get("asr", "model_id", MODEL_ID) or MODEL_ID
        self._penalty = float(cfg.get("asr", "repetition_penalty", REPETITION_PENALTY))
        self._model = None

    def params_hash(self) -> str:
        seed = f"{self._model_id}|{self._penalty}|{LANG}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def _load(self):
        if self._model is None:
            from mlx_audio.stt.generate import load_model

            self._model = load_model(self._model_id)
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        chunks: list[SpeechRegion],
        lang_hint: str | None = None,
        context: str | None = None,
    ) -> list[Segment]:
        """Transcribe the given spans. One Segment per span, at its real time.

        `context` is Qwen's native biasing input. It is powerful and therefore
        dangerous: given a global term list Whisper turned 資金流 into 紫金流.
        Callers should pass a list scoped to this video, not everything known.
        """
        if not chunks:
            raise NotImplementedError(
                "full-file transcription needs VAD chunking first: Qwen returns "
                "one untimed segment for its whole input, so every figure would "
                "land at t=0. See the ASR cross-check plan, 'Out of scope'."
            )

        from ..voice import read_slice

        model = self._load()
        lang = lang_hint or LANG
        out: list[Segment] = []
        for index, span in enumerate(chunks):
            try:
                audio = read_slice(Path(audio_path), span.start, span.duration)
                said = model.generate(
                    audio, language=lang, system_prompt=context,
                    repetition_penalty=self._penalty,
                )
            except Exception as exc:            # noqa: BLE001 - one span, not the video
                log.warning("asr.span_failed", start_s=span.start,
                            error=str(exc)[:200])
                continue
            text = (said.text or "").strip()
            if not text:
                continue
            out.append(Segment(
                id=index, start=span.start, end=span.end,
                text=text, lang=lang, confidence=None,
            ))
        return out
