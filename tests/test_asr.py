"""The engine is stubbed here. The model itself was measured by the bake-off on
real audio -- see docs/superpowers/specs/2026-08-11-asr-bake-off-results.md.
The repo already learned that stubbed tests cannot see real model behaviour, so
these cover only the wiring: span arithmetic, and that failures stay contained.
"""

import numpy as np
import pytest

from ytdigest.interfaces import SpeechRegion


class _FakeOut:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, audio, **kw):
        self.calls.append((audio, kw))
        return _FakeOut(f"heard{len(self.calls)}")


def _engine(monkeypatch, model=None):
    from ytdigest.asr.qwen3 import Qwen3ASRMLX

    cfg = type("C", (), {"get": lambda self, s, k, d=None: d})()
    eng = Qwen3ASRMLX(cfg)
    eng._model = model or _FakeModel()
    monkeypatch.setattr("ytdigest.voice.read_slice",
                        lambda *a, **k: np.zeros(16000, dtype=np.float32))
    return eng


def test_each_span_becomes_one_segment_at_its_real_timestamp(monkeypatch, tmp_path):
    """The span's own start/end must survive. Qwen returns a single segment
    spanning whatever it is given and reports it as 0.0-to-duration, so if the
    caller does not carry the real offsets every figure lands at t=0."""
    eng = _engine(monkeypatch)
    spans = [SpeechRegion(100.0, 116.0), SpeechRegion(250.5, 266.5)]
    segs = eng.transcribe(tmp_path / "a.wav", spans)

    assert [(s.start, s.end) for s in segs] == [(100.0, 116.0), (250.5, 266.5)]
    assert [s.text for s in segs] == ["heard1", "heard2"]


def test_confidence_stays_none(monkeypatch, tmp_path):
    """Qwen reports no score. Defaulting it to 1.0 would let the validator treat
    an unverified figure as verified -- the same trap auto-captions set."""
    eng = _engine(monkeypatch)
    segs = eng.transcribe(tmp_path / "a.wav", [SpeechRegion(0.0, 8.0)])
    assert segs[0].confidence is None


def test_a_failing_span_does_not_lose_the_others(monkeypatch, tmp_path):
    """One bad span must not cost the whole video its cross-check."""
    class _Flaky(_FakeModel):
        def generate(self, audio, **kw):
            self.calls.append(audio)
            if len(self.calls) == 1:
                raise RuntimeError("mlx blew up")
            return _FakeOut("ok")

    eng = _engine(monkeypatch, _Flaky())
    segs = eng.transcribe(tmp_path / "a.wav",
                          [SpeechRegion(0.0, 8.0), SpeechRegion(20.0, 28.0)])
    assert len(segs) == 1 and segs[0].start == 20.0


def test_full_file_mode_is_refused_rather_than_faked(monkeypatch, tmp_path):
    """Qwen returns one untimed segment for its whole input, so a full pass
    needs VAD chunking first. Returning a single 2.5-hour segment would put
    every view at t=0 and silently break voice ID."""
    eng = _engine(monkeypatch)
    with pytest.raises(NotImplementedError):
        eng.transcribe(tmp_path / "a.wav", [])


def test_the_refusal_is_permanent_not_retryable(monkeypatch, tmp_path):
    """runner's generic handler reads exc.error_class and defaults to retryable.
    A missing feature does not become present on the third attempt, and each
    retry re-pays the fetch stage for a channel that uploads twice a week.
    """
    from ytdigest.db import PERMANENT

    eng = _engine(monkeypatch)
    with pytest.raises(NotImplementedError) as caught:
        eng.transcribe(tmp_path / "a.wav", [])
    assert getattr(caught.value, "error_class", None) == PERMANENT
