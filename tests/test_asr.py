"""The engine is stubbed here. The model itself was measured by the bake-off on
real audio -- see docs/superpowers/specs/2026-08-11-asr-bake-off-results.md.
The repo already learned that stubbed tests cannot see real model behaviour, so
these cover only the wiring: span arithmetic, and that failures stay contained.
"""

import pytest
import torch

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


def _stub_slice():
    """The real `voice.read_slice` returns a mono torch tensor shaped (1, N)
    -- `torch.from_numpy(samples.copy()).unsqueeze(0)`, never a bare 1-D
    numpy array. A stub of the wrong type would let these tests pass while
    exercising a contract Qwen3ASRMLX.transcribe never actually sees. The
    real contract is pinned independently in
    tests/test_voice.py::test_read_slice_returns_a_mono_torch_tensor."""
    return torch.zeros((1, 16000), dtype=torch.float32)


def _engine(monkeypatch, model=None):
    from ytdigest.asr.qwen3 import Qwen3ASRMLX

    cfg = type("C", (), {"get": lambda self, s, k, d=None: d})()
    eng = Qwen3ASRMLX(cfg)
    eng._model = model or _FakeModel()
    monkeypatch.setattr("ytdigest.voice.read_slice",
                        lambda *a, **k: _stub_slice())
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


def test_every_span_failing_raises_rather_than_reporting_silence(monkeypatch, tmp_path):
    """A systematic failure -- missing torch, a non-16-bit wav, a truncated
    download -- fails every span. Returning [] here is indistinguishable from
    a video that was genuinely quiet throughout, and `_crosscheck_figures`
    would then write every ledger row `absent` (we listened, heard no number)
    when in truth nothing was ever heard. Must raise instead, so the caller's
    except leaves the rows NULL (never checked)."""
    class _AlwaysBroken(_FakeModel):
        def generate(self, audio, **kw):
            raise RuntimeError("mlx blew up")

    eng = _engine(monkeypatch, _AlwaysBroken())
    with pytest.raises(Exception):
        eng.transcribe(tmp_path / "a.wav",
                       [SpeechRegion(0.0, 8.0), SpeechRegion(20.0, 28.0)])


def test_all_spans_genuinely_silent_is_not_treated_as_a_failure(monkeypatch, tmp_path):
    """Every span transcribing cleanly to empty text is real silence -- the
    legitimate case this fix must not break. Must return [] quietly, not
    raise, so those rows are correctly written `absent`."""
    class _Silent(_FakeModel):
        def generate(self, audio, **kw):
            self.calls.append(audio)
            return _FakeOut("")

    eng = _engine(monkeypatch, _Silent())
    segs = eng.transcribe(tmp_path / "a.wav",
                          [SpeechRegion(0.0, 8.0), SpeechRegion(20.0, 28.0)])
    assert segs == []


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
