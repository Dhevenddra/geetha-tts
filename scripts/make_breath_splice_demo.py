r"""Listening demo: real breaths spliced into Anitha's and Arun's untouched studio speech.

    py -3.11 scripts\make_breath_splice_demo.py

12 Nov: the studio voices hold no breaths (D-055), and voice conversion of breathing
speakers into the two voices keeps the breath but ruins the voice (operator's ears). This
tries the remaining cheap idea before anything is trained: keep the studio speech exactly
as recorded and insert a real inhalation - cut from an IndicVoices-R clip at C4's
timestamps - before the first word or inside a natural pause.

A breath is unvoiced turbulence and carries little speaker identity; segments are kept only
if they are noise-like throughout (no voicing), so no word fragment of another speaker comes
along. Whether the result sounds like one person breathing is for the operator's ears.

Reads raw\ (Rasa train shards) and work\; writes work\breath_splice_demo\. No training data
is produced here.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geetha.audio.breath import HOP_SECONDS, frame_features  # noqa: E402

CALIBRATION = Path(r"E:\Geetha\work\breath_calibration")
ALIGNMENTS = Path(r"E:\Geetha\runs\c4_breath_timestamps\c4_alignments.json")
SEGMENTS = Path(r"E:\Geetha\work\studio_breath\segments.csv")
RASA_DIR = Path(r"E:\Geetha\raw\hf\rasa\Tamil")
OUT = Path(r"E:\Geetha\work\breath_splice_demo")
BREATH_BELOW_SPEECH_DB = (24.0, 30.0)   # an audible but ordinary inhalation
PER_VOICE = {"lead": 4, "gap": 6}
SEED = 20261112


def breath_bank(limit: int = 40) -> list[dict]:
    """Noise-like, voicing-free breath segments from high-SNR clips, one per speaker."""
    import soundfile as sf
    manifest = {c["id"]: c for c in
                json.loads((CALIBRATION / "manifest.json").read_text(encoding="utf-8"))["clips"]}
    found = []
    for record in json.loads(ALIGNMENTS.read_text(encoding="utf-8"))["clips"]:
        if record["label"] != "positive" or manifest[record["id"]]["snr"] < 55:
            continue
        gaps = [g for g in record["gaps"] if g["has_breath"] and 0.35 <= g["seconds"] <= 1.0]
        if not gaps:
            continue
        wave, rate = sf.read(str(CALIBRATION / "clips" / f"{record['id']}.flac"), dtype="float32")
        frames = frame_features(wave)
        energy = frames["energy_db"]
        floor, peak = float(np.percentile(energy, 2)), float(np.percentile(energy, 95))
        for gap in gaps:
            first = int((gap["start"] + 0.06) / HOP_SECONDS)
            last = int((gap["end"] - 0.06) / HOP_SECONDS)
            if last - first < 20:
                continue
            window = slice(first, last)
            breathy = ((energy[window] >= floor + 8) & (energy[window] <= peak - 12)
                       & (frames["centroid"][window] >= 1200) & (frames["low"][window] <= 0.15))
            padded = np.concatenate([[False], breathy, [False]])
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            runs = sorted(zip(edges[::2], edges[1::2]), key=lambda r: r[0] - r[1])
            if not runs or runs[0][1] - runs[0][0] < 20:      # at least 200 ms of pure breath
                continue
            start, end = runs[0]
            a = int((first + start) * HOP_SECONDS * rate)
            b = int((first + end) * HOP_SECONDS * rate)
            found.append({"speaker": manifest[record["id"]]["speaker_id"], "wave": wave[a:b].copy(),
                          "rate": rate, "seconds": (b - a) / rate,
                          "high": float(frames["high"][window][start:end].mean())})
    found.sort(key=lambda row: -row["high"] * min(row["seconds"], 0.6))
    bank, seen = [], set()
    for row in found:
        if row["speaker"] not in seen:
            seen.add(row["speaker"])
            bank.append(row)
    return bank[:limit]


def to_rate(wave: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return wave
    grid = np.arange(int(len(wave) * target / source)) * (source / target)
    out = np.interp(grid, np.arange(len(wave)), wave)
    if target > source:                      # remove imaging above the source's band
        taps = np.arange(-120, 121)
        kernel = np.sinc(2 * (0.47 * source) / target * taps) * np.blackman(len(taps))
        out = np.convolve(out, kernel / kernel.sum(), "same")
    return out.astype(np.float32)


def faded(wave: np.ndarray, rate: int, seconds: float = 0.04) -> np.ndarray:
    n = min(int(seconds * rate), len(wave) // 2)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    out = wave.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def speech_rms(wave: np.ndarray, rate: int) -> float:
    hop = int(0.02 * rate)
    frames = wave[: len(wave) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    return float(np.sqrt((rms[rms >= np.percentile(rms, 60)] ** 2).mean()))


def splice(clip: np.ndarray, rate: int, breath: dict, kind: str, start: float, end: float,
           below_db: float) -> tuple[np.ndarray, float]:
    """Returns the new clip and the second at which the breath begins."""
    piece = faded(to_rate(breath["wave"], breath["rate"], rate), rate)
    piece = piece[: int(0.6 * rate)] if len(piece) > int(0.6 * rate) else piece
    piece = faded(piece, rate)
    target = speech_rms(clip, rate) * 10 ** (-below_db / 20)
    piece = piece * (target / (np.sqrt((piece ** 2).mean()) + 1e-9))
    quiet = clip[int(start * rate): int(end * rate)]
    if kind == "lead":
        onset = int(max(0.0, end - 0.05) * rate)
        room = quiet[: int(0.05 * rate)] if len(quiet) >= int(0.05 * rate) else np.zeros(int(0.05 * rate), np.float32)
        out = np.concatenate([room, piece, room, clip[onset:]])
        return out.astype(np.float32), len(room) / rate
    a, b = int((start + 0.07) * rate), int((end - 0.07) * rate)
    room = clip[a: a + int(0.04 * rate)]
    out = np.concatenate([clip[:a], room, piece, room, clip[b:]])
    return out.astype(np.float32), (a + len(room)) / rate


def main() -> int:
    import pyarrow.parquet as pq
    import soundfile as sf

    rng = np.random.default_rng(SEED)
    bank = breath_bank()
    print(f"breath bank: {len(bank)} segments, {min(b['seconds'] for b in bank):.2f}-"
          f"{max(b['seconds'] for b in bank):.2f} s")
    rows = [r for r in csv.DictReader(SEGMENTS.open(encoding="utf-8"))
            if r["style"] in ("CONV", "BOOK") and 3.0 <= float(r["clip_s"]) <= 9.0]
    OUT.mkdir(parents=True, exist_ok=True)
    sheet, number = [], 0
    for voice in ("anitha", "arun"):
        mine = [r for r in rows if r["speaker"] == voice]
        pools = {
            "lead": [r for r in mine if r["kind"] == "lead"],
            "gap": [r for r in mine if r["kind"] == "gap" and 0.28 <= float(r["dur"]) <= 0.7
                    and int(r["burst_frames"]) < 5],
        }
        used: set[str] = set()
        for kind, count in PER_VOICE.items():
            pool = [r for r in pools[kind] if r["clip"] not in used]
            for i in rng.choice(len(pool), size=count, replace=False):
                r = pool[int(i)]
                used.add(r["clip"])
                record = pq.ParquetFile(str(RASA_DIR / r["shard"])).read().slice(int(r["row"]), 1).to_pylist()[0]
                assert record["filename"] == r["clip"]
                clip, rate = sf.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
                if clip.ndim > 1:
                    clip = clip.mean(axis=1)
                breath = bank[int(rng.integers(len(bank)))]
                below = float(rng.uniform(*BREATH_BELOW_SPEECH_DB))
                out, at = splice(clip, rate, breath, kind, float(r["start"]), float(r["end"]), below)
                number += 1
                name = f"{number:02d}_{voice}_{'before_first_word' if kind == 'lead' else 'mid_pause'}"
                sf.write(str(OUT / f"{name}_breath.wav"), out, rate)
                sf.write(str(OUT / f"{name}_original.wav"), clip, rate)
                sheet.append(f"{name}_breath.wav\tbreath at {at:.2f}s\t{below:.0f} dB under speech\t{record['text']}")
    (OUT / "LISTEN.tsv").write_text("\n".join(sheet), encoding="utf-8")
    print(f"wrote {number} pairs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
