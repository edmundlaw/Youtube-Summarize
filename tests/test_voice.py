"""Speaker identification. Every test here guards against crediting one
person with another person's call, which is the failure this module exists to
prevent -- and which shipped undetected before it existed.
"""

from __future__ import annotations

import pathlib
import struct
import tempfile
import wave

import pytest

from ytdigest import db as D
from ytdigest import voice


@pytest.fixture
def conn():
    path = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    c = D.open_db(path, pathlib.Path("migrations"))
    c.execute("INSERT INTO channels (id,title,added_at) VALUES ('UC1','c',?)",
              (D.now_iso(),))
    c.execute("INSERT INTO videos (id,channel_id,title,published_at,discovered_at,"
              "status) VALUES ('v1','UC1','t','2026-01-01','2026-01-01','done')")
    return c


def add_print(conn, speaker, vector):
    conn.execute(
        "INSERT INTO voiceprints (speaker,embedding,dim,model,n_clips,total_s,"
        "source_note,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (speaker, voice._pack(voice._l2(vector)), len(vector), voice.MODEL,
         100, 400.0, "test", D.now_iso(), D.now_iso()))


def unit(index, dim=voice.DIM):
    v = [0.0] * dim
    v[index] = 1.0
    return v


# --- storage round-trip -----------------------------------------------------

def test_embedding_survives_the_database(conn):
    """A voiceprint that changes on the way to disk silently stops matching."""
    import numpy as np
    original = voice._l2([0.3, -0.7, 0.1, 0.9] * 48)
    add_print(conn, "someone", original)
    restored = voice.voiceprints(conn)["someone"]
    assert np.allclose(original, restored, atol=1e-6)


def test_voiceprints_from_another_model_are_not_returned(conn):
    """Embeddings are not comparable across models. Silently matching against
    a stale one would produce confident nonsense."""
    conn.execute(
        "INSERT INTO voiceprints (speaker,embedding,dim,model,n_clips,total_s,"
        "source_note,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("old", voice._pack(unit(0)), voice.DIM, "some-older-model",
         100, 400.0, "test", D.now_iso(), D.now_iso()))
    assert voice.voiceprints(conn) == {}


# --- the refusals -----------------------------------------------------------

def test_no_voiceprints_means_no_attribution(conn):
    assert voice.identify(conn, pathlib.Path("/nonexistent.wav"),
                          [{"start": 0.0, "end": 10.0}], 0.6, 0.08) == []


def test_short_segments_are_refused_before_any_audio_is_read(conn):
    """Interjections are where cross-talk lives; they are also too short to
    identify. Refusing them costs almost nothing."""
    add_print(conn, "KC", unit(0))
    rows = voice.identify(conn, pathlib.Path("/nonexistent.wav"),
                          [{"start": 0.0, "end": 0.5}], 0.6, 0.08)
    assert rows[0].speaker is None
    assert "too short" in rows[0].note


def test_unreadable_audio_yields_no_speaker_not_a_wrong_one(conn):
    add_print(conn, "KC", unit(0))
    rows = voice.identify(conn, pathlib.Path("/nonexistent.wav"),
                          [{"start": 0.0, "end": 30.0}], 0.6, 0.08)
    assert rows[0].speaker is None


# --- the two gates ----------------------------------------------------------

def _identify_with(conn, monkeypatch, vector, threshold=0.6, margin=0.08):
    import numpy as np
    monkeypatch.setattr(voice, "embed_slices",
                        lambda *a, **k: {0: voice._l2(np.array(vector, dtype="float32"))})
    return voice.identify(conn, pathlib.Path("x.wav"),
                          [{"start": 0.0, "end": 30.0}], threshold, margin)[0]


def test_voice_matching_nobody_enrolled_is_left_unattributed(conn, monkeypatch):
    """A guest, an advert or a clip must not be forced onto the nearest host."""
    add_print(conn, "KC", unit(0))
    add_print(conn, "Eugene", unit(1))
    row = _identify_with(conn, monkeypatch, unit(5))       # orthogonal to both
    assert row.speaker is None
    assert "no enrolled voice matched" in row.note


def test_a_voice_between_two_hosts_is_refused_as_cross_talk(conn, monkeypatch):
    """Two people talking at once lands between their voiceprints. Picking the
    marginally closer one is how a disagreement gets attributed backwards."""
    add_print(conn, "KC", unit(0))
    add_print(conn, "Eugene", unit(1))
    blend = [0.0] * voice.DIM
    blend[0], blend[1] = 0.71, 0.70                        # near-equal to both
    row = _identify_with(conn, monkeypatch, blend)
    assert row.speaker is None
    assert "ambiguous" in row.note


def test_a_clear_match_is_attributed(conn, monkeypatch):
    add_print(conn, "KC", unit(0))
    add_print(conn, "Eugene", unit(1))
    row = _identify_with(conn, monkeypatch, unit(0))
    assert row.speaker == "KC"
    assert row.score == pytest.approx(1.0, abs=1e-5)


def test_threshold_is_honoured(conn, monkeypatch):
    """The calibrated 0.60 must actually bind: a different host on this corpus
    scores ~0.40 against another's voiceprint, so a lower gate admits them."""
    add_print(conn, "KC", unit(0))
    vector = [0.0] * voice.DIM
    vector[0], vector[7] = 0.45, 0.89                      # cos ~0.45 to KC
    assert _identify_with(conn, monkeypatch, vector, threshold=0.6).speaker is None
    assert _identify_with(conn, monkeypatch, vector, threshold=0.3).speaker == "KC"


# --- enrolment refuses thin evidence ---------------------------------------

def test_enrolment_refuses_too_little_audio(conn, monkeypatch):
    """A voiceprint from a handful of windows is dominated by whatever noise
    those windows happened to contain."""
    monkeypatch.setattr(voice, "embed_slices", lambda *a, **k: {0: unit(0)})
    monkeypatch.setattr(voice, "audio_duration_s", lambda p: 600.0)
    with pytest.raises(ValueError, match="at least 10"):
        voice.enroll(conn, "KC", [pathlib.Path("a.wav")], "test")


# --- audio is always deleted ------------------------------------------------

def _write_wav(path, seconds=1.0, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{int(rate * seconds)}h",
                                       *([1000] * int(rate * seconds))))


def test_audio_is_deleted_even_when_the_caller_raises(monkeypatch, tmp_path):
    """Audio dwarfs everything else stored here. Deleting only on success would
    let a crashed run fill the disk."""
    wav = tmp_path / "vid1.wav"
    monkeypatch.setattr("ytdigest.sources.youtube.fetch_audio",
                        lambda vid, d: (_write_wav(wav), wav)[1])
    with pytest.raises(RuntimeError):
        with voice.audio_for("vid1", tmp_path):
            assert wav.exists()
            raise RuntimeError("stage blew up")
    assert not wav.exists()


def test_audio_is_deleted_on_success(monkeypatch, tmp_path):
    wav = tmp_path / "vid2.wav"
    monkeypatch.setattr("ytdigest.sources.youtube.fetch_audio",
                        lambda vid, d: (_write_wav(wav), wav)[1])
    with voice.audio_for("vid2", tmp_path) as path:
        assert path.exists()
    assert not wav.exists()
    assert list(tmp_path.glob("vid2.*")) == []


# --- reading slices ---------------------------------------------------------

def test_read_slice_seeks_rather_than_loading_everything(tmp_path):
    wav = tmp_path / "a.wav"
    _write_wav(wav, seconds=10.0)
    assert voice.audio_duration_s(wav) == pytest.approx(10.0, abs=0.01)
    chunk = voice.read_slice(wav, 5.0, 2.0)
    assert chunk is not None
    assert chunk.shape[-1] == pytest.approx(2 * voice.SAMPLE_RATE, abs=2)


def test_read_slice_past_the_end_returns_nothing(tmp_path):
    wav = tmp_path / "b.wav"
    _write_wav(wav, seconds=2.0)
    assert voice.read_slice(wav, 60.0, 2.0) is None


# --- voice outranks the model ----------------------------------------------

def _store_view(conn, speaker, start_s=100.0):
    from ytdigest.views import View, store_views
    view = View(
        speaker=speaker, start_s=start_s, instrument_raw="NVIDIA",
        instrument=None, asset_class=None, direction="long", thesis="t",
        reasoning=None, level_type="entry", level_value=120.0, level_unit="usd",
        ledger_id=None, level_verified=False, horizon="weeks",
        conviction="medium", entry_basis="immediate", condition=None,
        stance="bullish",
    )
    store_views(conn, {"id": "v1", "channel_id": "UC1",
                       "published_at": "2026-01-01T00:00:00+00:00"},
                [view], None, "v3")
    return conn.execute("SELECT speaker, attribution FROM views "
                        "ORDER BY id DESC LIMIT 1").fetchone()


def test_voice_overrides_the_models_guess(conn):
    """The exact failure Edmund caught: the model put a NVIDIA call in KC's
    mouth at a timestamp where the voice was not KC. A measurement must beat
    an inference, so the stored view carries the voice's answer."""
    conn.execute(
        "INSERT INTO segment_speakers (video_id,start_s,end_s,speaker,score,"
        "margin,model,created_at) VALUES ('v1',90.0,130.0,'Eugene',0.81,0.2,?,?)",
        (voice.MODEL, D.now_iso()))
    row = _store_view(conn, speaker="羅家聰 (KC)")          # model said KC
    assert row["speaker"] == "Eugene"
    assert row["attribution"] == "voice"


def test_unidentified_voice_drops_the_name_rather_than_trusting_the_model(conn):
    """Once identification has run, its refusal is the answer. Falling back to
    the model's guess here is precisely how a wrong name reaches the record."""
    conn.execute(
        "INSERT INTO segment_speakers (video_id,start_s,end_s,speaker,score,"
        "margin,model,created_at) VALUES ('v1',90.0,130.0,NULL,0.35,0.01,?,?)",
        (voice.MODEL, D.now_iso()))
    row = _store_view(conn, speaker="羅家聰 (KC)")
    assert row["speaker"] is None
    assert row["attribution"] == "none"


def test_without_identification_the_name_is_marked_as_guessed(conn):
    """No voice data is not the same as a refusal. The name is kept so the
    record is not thrown away, but flagged so it is never confused with one
    that was measured."""
    row = _store_view(conn, speaker="羅家聰 (KC)")
    assert row["speaker"] == "羅家聰 (KC)"
    assert row["attribution"] == "guessed"


# --- enrolment purity -------------------------------------------------------

def _two_speakers(shared=0.816):
    """Two voiceprints ~0.40 apart, which is what different hosts actually
    measure at on this corpus. Orthogonal synthetic vectors would make these
    tests pass for the wrong reason: with truly independent speakers the mean
    lands nowhere near either, which is not the situation being guarded
    against. The shared component stands for the common language, subject and
    recording chain that put real hosts so close together."""
    import numpy as np
    shared_axis = np.zeros(voice.DIM, dtype="float32"); shared_axis[0] = 1.0
    own_a = np.zeros(voice.DIM, dtype="float32"); own_a[1] = 1.0
    own_b = np.zeros(voice.DIM, dtype="float32"); own_b[2] = 1.0
    return (voice._l2(shared * shared_axis + own_a),
            voice._l2(shared * shared_axis + own_b))


def _windows(centre, n, seed, sigma=0.05):
    """Sigma chosen so within-speaker similarity lands at ~0.82, matching the
    measured spread of real enrolment audio."""
    import numpy as np
    rng = np.random.default_rng(seed)
    return [voice._l2(centre + rng.normal(0, sigma, voice.DIM).astype("float32"))
            for _ in range(n)]


def test_purity_flags_a_voiceprint_averaged_over_two_people(conn, monkeypatch):
    """The worst input error possible here: enrolling from a video that is not
    actually solo. It fails silently, producing a voiceprint that matches
    neither person well and attributing whichever of them scores higher."""
    a, b = _two_speakers()
    clips = _windows(a, 30, 0) + _windows(b, 30, 1)
    monkeypatch.setattr(voice, "embed_slices", lambda *x, **k: dict(enumerate(clips)))
    monkeypatch.setattr(voice, "audio_duration_s", lambda p: 900.0)
    result = voice.enroll(conn, "mixed", [pathlib.Path("a.wav")], "test")
    assert result["purity"] < voice.MIN_PURITY
    assert "more than one person" in voice.purity_warning(result)


def test_purity_is_quiet_for_a_genuinely_solo_source(conn, monkeypatch):
    a, _ = _two_speakers()
    clips = _windows(a, 60, 2)
    monkeypatch.setattr(voice, "embed_slices", lambda *x, **k: dict(enumerate(clips)))
    monkeypatch.setattr(voice, "audio_duration_s", lambda p: 900.0)
    result = voice.enroll(conn, "solo", [pathlib.Path("a.wav")], "test")
    assert result["purity"] >= voice.MIN_PURITY
    assert voice.purity_warning(result) is None
