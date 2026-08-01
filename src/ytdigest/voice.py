"""Who actually said it — speaker identification from audio.

YouTube auto-captions carry no speaker labels. On a three-host show the model
summarising them had no way to know who spoke, so every name it printed was a
guess. A guessed attribution is worse than none: the entire purpose of the
track record is judging named individuals, and a wrong name silently credits
one person with another's call.

The information exists only in the audio, so that is where we get it. Not full
diarization — a narrower and more reliable question: *does this segment sound
like one of the people we already have a voiceprint for?* Enrolment is free for
any host with a solo video, since every second of it is definitionally them.

The governing rule matches the rest of the pipeline: **refuse rather than
guess.** A segment is attributed only when it clears an absolute similarity
threshold *and* beats the runner-up by a margin. Everything else stays
unattributed, which is an honest and useful answer.

Deliberately NOT done here:
* No clustering of unknown voices into "speaker 2". A cluster with no name
  cannot be scored, and inventing one invites it being matched to a person by
  eye later.
* No attribution of overlapping speech. Cross-talk is where every speaker-ID
  method degrades, and these shows have a lot of it.
* Audio is never kept. It is an order of magnitude larger than everything else
  this project stores, and it has no value once embedded.
"""

from __future__ import annotations

import contextlib
import struct
from dataclasses import dataclass
from pathlib import Path

from .db import now_iso, transaction

#: Embeddings from different models are not comparable. Stored alongside every
#: voiceprint so a model change invalidates rather than silently corrupts.
MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16_000
DIM = 192

#: A segment shorter than this does not carry enough voice to identify. Short
#: interjections ("係啊", "唔係") are exactly where cross-talk lives, so
#: refusing them costs little and avoids the least reliable attributions.
MIN_SEGMENT_S = 1.5

#: Longest slice fed to the encoder. Beyond a few seconds accuracy stops
#: improving and memory does not.
MAX_SEGMENT_S = 8.0

#: Silence and background music produce confident-looking nonsense embeddings.
MIN_RMS = 0.005

#: Calibrated on this corpus, not taken from a paper. A voiceprint was built
#: from two solo KC videos and scored against (a) held-out KC and (b) a
#: different host on the same channel style:
#:
#:     threshold   KC accepted   different speaker accepted
#:      0.50           84%              4%
#:      0.55           83%              0%
#:      0.60           82%              0%
#:
#: 0.60 is chosen over 0.55 because it costs one point of recall and buys
#: headroom: false positives are the failure that corrupts a track record,
#: missed attributions merely leave a segment unlabelled. Note that different
#: speakers here score ~0.40 against each other, far above the ~0.1 typical of
#: unrelated audio -- same language, same subject, similar recording chain -- so
#: a threshold borrowed from published benchmarks would have been far too low.
DEFAULT_THRESHOLD = 0.60

#: Best match must beat the runner-up by this much. Two enrolled voices scoring
#: near-equally is what overlapping speech looks like in embedding space, and
#: cross-talk is precisely where attribution must not be attempted.
DEFAULT_MARGIN = 0.08

_encoder = None


def encoder():
    """Load once per process. ~30 s cold, then cached."""
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _encoder = EncoderClassifier.from_hparams(
            source=MODEL, savedir=str(Path.home() / ".cache" / "ytdigest" / "ecapa"),
        )
    return _encoder


@dataclass
class Attribution:
    start_s: float
    end_s: float
    speaker: str | None
    score: float | None
    margin: float | None
    note: str


def _pack(vector) -> bytes:
    return struct.pack(f"<{len(vector)}f", *[float(x) for x in vector])


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _l2(vector):
    import numpy as np
    vector = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def audio_duration_s(wav_path: Path) -> float:
    import wave as wavelib
    try:
        with contextlib.closing(wavelib.open(str(wav_path))) as handle:
            return handle.getnframes() / float(handle.getframerate() or SAMPLE_RATE)
    except Exception:
        return 0.0


def read_slice(wav_path: Path, start_s: float, duration_s: float):
    """Read one slice without loading the whole file.

    A 2.5-hour 16 kHz wav is ~570 MB as float32. This machine shares 16 GB with
    a dozen other services, so slices are seeked to rather than sliced out of a
    resident array.

    Read with the standard library rather than torchaudio: `fetch_audio` always
    writes 16 kHz mono 16-bit PCM, `wave` seeks it exactly, and torchaudio moved
    its top-level load/info out from under us between releases. One less moving
    part in the layer that must not silently return the wrong audio.
    """
    import wave as wavelib

    import numpy as np
    import torch

    frames = int(duration_s * SAMPLE_RATE)
    if frames <= 0:
        return None
    try:
        with contextlib.closing(wavelib.open(str(wav_path))) as handle:
            if handle.getsampwidth() != 2:
                return None
            rate = handle.getframerate() or SAMPLE_RATE
            channels = handle.getnchannels() or 1
            offset = int(start_s * rate)
            if offset >= handle.getnframes():
                return None
            handle.setpos(offset)
            raw = handle.readframes(int(duration_s * rate))
    except Exception:
        return None
    if not raw:
        return None

    samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:                       # stereo slipped through
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:                # fetch_audio pins 16 kHz; be safe
        target = int(len(samples) * SAMPLE_RATE / rate)
        if target <= 0:
            return None
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target),
            np.arange(len(samples)), samples).astype("float32")
    return torch.from_numpy(samples.copy()).unsqueeze(0)


def embed_slices(wav_path: Path, spans: list[tuple[float, float]],
                 batch_size: int = 16):
    """Embed each (start, duration) span. Returns {index: unit vector}.

    Spans that are silent, unreadable or too short are simply absent from the
    result rather than embedded badly — a missing attribution is recoverable,
    a wrong one is not.
    """
    import numpy as np
    import torch

    model = encoder()
    out: dict[int, "np.ndarray"] = {}
    batch: list[tuple[int, "torch.Tensor"]] = []

    def flush():
        if not batch:
            return
        width = max(w.shape[-1] for _, w in batch)
        padded = torch.zeros(len(batch), width)
        lengths = torch.zeros(len(batch))
        for row, (_, wave) in enumerate(batch):
            padded[row, :wave.shape[-1]] = wave.squeeze(0)
            lengths[row] = wave.shape[-1] / width
        with torch.no_grad():
            vectors = model.encode_batch(padded, lengths).squeeze(1).cpu().numpy()
        for row, (index, _) in enumerate(batch):
            out[index] = _l2(vectors[row])
        batch.clear()

    for index, (start, duration) in enumerate(spans):
        wave = read_slice(wav_path, start, min(duration, MAX_SEGMENT_S))
        if wave is None:
            continue
        if float(wave.pow(2).mean().sqrt()) < MIN_RMS:      # silence or music
            continue
        batch.append((index, wave))
        if len(batch) >= batch_size:
            flush()
    flush()
    return out


# --- enrolment --------------------------------------------------------------

def enroll(conn, speaker: str, wav_paths: list[Path], source_note: str,
           window_s: float = 4.0, stride_s: float = 6.0,
           max_windows: int = 240) -> dict:
    """Build a voiceprint by averaging windows of known-speaker audio.

    Averaging many windows is what makes this robust: any single window may
    catch a cough, a jingle or a co-host's interjection, but the mean over a
    few hundred is dominated by the person who is actually talking throughout.
    """
    import numpy as np

    vectors, total_s = [], 0.0
    for path in wav_paths:
        duration = audio_duration_s(path)
        if duration <= window_s:
            continue
        spans = [(t, window_s)
                 for t in np.arange(0.0, max(duration - window_s, 0.0), stride_s)]
        if len(spans) > max_windows:                 # even sample, not a prefix
            step = len(spans) / max_windows
            spans = [spans[int(i * step)] for i in range(max_windows)]
        found = embed_slices(path, spans)
        vectors.extend(found.values())
        total_s += len(found) * window_s

    if len(vectors) < 10:
        raise ValueError(
            f"only {len(vectors)} usable windows for {speaker}; "
            "need at least 10 to build a voiceprint worth trusting")

    stacked = np.stack(vectors)
    mean = _l2(np.mean(stacked, axis=0))
    purity = float((stacked @ mean > PURITY_SIMILARITY).mean())
    with transaction(conn):
        conn.execute(
            "INSERT INTO voiceprints (speaker,embedding,dim,model,n_clips,"
            " total_s,source_note,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(speaker) DO UPDATE SET embedding=excluded.embedding, "
            "  dim=excluded.dim, model=excluded.model, n_clips=excluded.n_clips, "
            "  total_s=excluded.total_s, source_note=excluded.source_note, "
            "  updated_at=excluded.updated_at",
            (speaker, _pack(mean), DIM, MODEL, len(vectors), total_s,
             source_note, now_iso(), now_iso()),
        )
    return {"speaker": speaker, "windows": len(vectors), "seconds": total_s,
            "purity": purity}


#: Contamination check, calibrated on real embeddings. Two earlier ideas were
#: tried and both failed on measurement rather than in theory:
#:
#:   * share of windows above the 0.60 identification threshold -- useless,
#:     because with two speakers the mean sits between them and both score well
#:     above it (65% for a deliberate 50/50 mix, against 82% for genuine solo).
#:   * two-means centroid separation -- worse than useless: the deliberately
#:     mixed source scored *highest* (0.550) and a real two-person interview
#:     *lowest* (0.180). Within one recording the dominant variation is acoustic
#:     conditions, not speaker identity, so the clusters are noise.
#:
#: What does separate, measured at similarity 0.70:
#:
#:     genuine solo        52%, 74%, 78%, 82%
#:     50/50 two speakers  32%
#:     two-person interview 2%
#:
#: So: share of enrolment windows within 0.70 of their own mean. This is a
#: warning, not a proof -- the gap between 32% and 52% is real but not wide,
#: and a genuinely solo recording with heavy music or phone audio could dip
#: into it. It is checked because titles are not reliable: 「文錦期權譜」 lists
#: one host and another name presents some episodes.
PURITY_SIMILARITY = 0.70
MIN_PURITY = 0.45


def purity_warning(result: dict) -> str | None:
    """Flag a voiceprint that looks like it was averaged over two people.

    This is the worst input error possible here: it does not fail loudly, it
    produces a voiceprint that matches neither person well and attributes
    whichever of them happens to score higher. Titles are not reliable enough
    to catch it -- 「文錦期權譜」 lists one host but another name presents some
    episodes -- so the check is on the audio itself.
    """
    purity = result.get("purity")
    if purity is None or purity >= MIN_PURITY:
        return None
    return (f"only {purity:.0%} of windows match the resulting voiceprint "
            f"(expect >{MIN_PURITY:.0%} for one speaker). The source audio "
            "probably contains more than one person — check the video for a "
            "co-host or guest before trusting this voiceprint.")


def voiceprints(conn) -> dict[str, list[float]]:
    return {r["speaker"]: _unpack(r["embedding"])
            for r in conn.execute(
                "SELECT speaker, embedding FROM voiceprints WHERE model = ?",
                (MODEL,))}


# --- identification ---------------------------------------------------------

def identify(conn, wav_path: Path, segments: list[dict],
             threshold: float, margin: float) -> list[Attribution]:
    """Attribute each caption segment to an enrolled speaker, or to nobody.

    Two gates, both required. The absolute `threshold` rejects a voice that
    resembles nobody we know — a guest, a clip, an advert. The `margin` rejects
    a voice that resembles two people almost equally, which is what overlapping
    speech looks like in embedding space. Failing either yields None, and None
    is a perfectly good answer.
    """
    import numpy as np

    prints = voiceprints(conn)
    if not prints:
        return []
    names = list(prints)
    matrix = np.stack([_l2(prints[n]) for n in names])

    usable, spans = [], []
    results: list[Attribution] = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment.get("end") or start)
        if end - start < MIN_SEGMENT_S:
            results.append(Attribution(start, end, None, None, None,
                                       "segment too short to identify"))
            continue
        usable.append((len(results), start, end))
        spans.append((start, end - start))
        results.append(Attribution(start, end, None, None, None, "pending"))

    found = embed_slices(wav_path, spans)
    for order, (slot, start, end) in enumerate(usable):
        vector = found.get(order)
        if vector is None:
            results[slot] = Attribution(start, end, None, None, None,
                                        "silent or unreadable audio")
            continue
        scores = matrix @ vector
        best = int(np.argmax(scores))
        top = float(scores[best])
        second = float(np.sort(scores)[-2]) if len(scores) > 1 else -1.0
        gap = top - second
        if top < threshold:
            results[slot] = Attribution(start, end, None, top, gap,
                                        f"no enrolled voice matched ({top:.2f})")
        elif gap < margin:
            results[slot] = Attribution(
                start, end, None, top, gap,
                f"ambiguous between {names[best]} and a close second ({gap:.2f})")
        else:
            results[slot] = Attribution(start, end, names[best], top, gap, "ok")
    return results


def store_attributions(conn, video_id: str, rows: list[Attribution]) -> int:
    with transaction(conn):
        conn.execute("DELETE FROM segment_speakers WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO segment_speakers (video_id,start_s,end_s,speaker,score,"
            " margin,model,created_at) VALUES (?,?,?,?,?,?,?,?)",
            [(video_id, r.start_s, r.end_s, r.speaker, r.score, r.margin,
              MODEL, now_iso()) for r in rows],
        )
    return sum(1 for r in rows if r.speaker)


def speaker_at(conn, video_id: str, start_s: float) -> str | None:
    """Who was speaking at a timestamp, per stored identification."""
    row = conn.execute(
        "SELECT speaker FROM segment_speakers WHERE video_id = ? AND start_s <= ? "
        "AND end_s >= ? ORDER BY start_s DESC LIMIT 1",
        (video_id, start_s, start_s),
    ).fetchone()
    return row["speaker"] if row else None


# --- audio lifecycle --------------------------------------------------------

@contextlib.contextmanager
def audio_for(video_id: str, dest_dir: Path):
    """Download audio, hand it over, and always delete it.

    Audio dwarfs everything else this project keeps -- a 2.5-hour show is
    ~570 MB as 16 kHz wav against ~36 KB of transcript -- and once embedded it
    has no further use. Deleting in `finally` means a crashed or interrupted
    identification does not quietly fill the disk.
    """
    from .sources.youtube import fetch_audio

    path = None
    try:
        path = fetch_audio(video_id, dest_dir)
        yield path
    finally:
        for leftover in dest_dir.glob(f"{video_id}.*"):
            if leftover.suffix.lower() in {".wav", ".m4a", ".webm", ".opus",
                                           ".mp3", ".part", ".ytdl"}:
                with contextlib.suppress(OSError):
                    leftover.unlink()
