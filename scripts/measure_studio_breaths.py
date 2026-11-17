r"""Do Anitha's and Arun's studio recordings contain audible breaths at all? (D-054 item 2)

    py -3.11 scripts\measure_studio_breaths.py scan      # every Rasa train shard -> segments.csv
    py -3.11 scripts\measure_studio_breaths.py control   # same statistic on C4's labelled gaps
    py -3.11 scripts\measure_studio_breaths.py listen    # cut clips for the operator's ears

Why this is not the D-030 detector
----------------------------------
The D-030 model was fitted on 8 kHz telephone speech, where nothing exists above 4 kHz.
Rasa is 48 kHz studio audio; its band fractions are a different world, so that model's
weights mean nothing here. This script instead asks a physical question that needs no
fitted weights: inside the non-speech parts of a clip (before the first word, between
words, after the last), is there an *isolated burst of noise* - at least 120 ms long, at
least 8 dB over the clip's own floor, separated from speech by near-silence? That is what
an inhalation looks like. It is also what a lip smack or a page turn looks like, so the
count is an upper bound and **the operator's ears are the ground truth** (`listen`).

`control` runs the same burst statistic on C4's 800 calibration clips, where each gap is
labelled breath / no breath, so the statistic is tested against truth before it is trusted.

Reads `raw\` only; writes under `work\studio_breath\`. Train shards only (test is frozen).
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geetha.audio.breath import HOP_SECONDS, SAMPLE_RATE, frame_features  # noqa: E402

RASA = r"E:\Geetha\raw\hf\rasa\Tamil\train-*.parquet"
OUT = Path(r"E:\Geetha\work\studio_breath")
CALIBRATION = Path(r"E:\Geetha\work\breath_calibration")
ALIGNMENTS = Path(r"E:\Geetha\runs\c4_breath_timestamps\c4_alignments.json")

SPEECH_BELOW_PEAK_DB = 30.0   # a frame within this of the clip's 95th percentile is speech
ACTIVE_OVER_FLOOR_DB = 8.0    # a non-speech frame this far over the floor is "something"
GUARD_FRAMES = 3              # 30 ms of near-silence must separate a burst from speech
MIN_BURST_FRAMES = 12         # 120 ms: shorter is a click or a lip smack, not an inhalation
VOICE = {"Female": "anitha", "Male": "arun"}

FIELDS = ["clip", "speaker", "style", "shard", "row", "clip_s", "kind", "start", "end", "dur",
          "floor_dbfs", "zero_frac", "burst_frames", "burst_start", "burst_db", "burst_high",
          "burst_flat", "burst_cent", "isolated"]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def best_burst(frames: dict, first: int, last: int, floor: float, clip_frames: int,
               guard_left: bool = True, guard_right: bool = True) -> dict:
    """The longest above-floor run inside one non-speech segment, and whether it stands
    apart from the speech on either side."""
    excess = frames["energy_db"][first:last] - floor
    best = {"burst_frames": 0, "burst_start": 0.0, "burst_db": 0.0, "burst_high": 0.0,
            "burst_flat": 0.0, "burst_cent": 0.0, "isolated": 0}
    for start, end in _runs(excess >= ACTIVE_OVER_FLOOR_DB):
        if end - start <= best["burst_frames"]:
            continue
        left_ok = (not guard_left) or first == 0 or start >= GUARD_FRAMES
        right_ok = (not guard_right) or last >= clip_frames or (last - first) - end >= GUARD_FRAMES
        window = slice(first + start, first + end)
        best = {
            "burst_frames": int(end - start),
            "burst_start": round((first + start) * HOP_SECONDS, 3),
            "burst_db": round(float(excess[start:end].mean()), 2),
            "burst_high": round(float(frames["high"][window].mean()), 4),
            "burst_flat": round(float(frames["flat"][window].mean()), 4),
            "burst_cent": round(float(frames["centroid"][window].mean()), 1),
            "isolated": int(left_ok and right_ok),
        }
    return best


def segments(wave16: np.ndarray) -> tuple[list[dict], dict]:
    """Non-speech segments of one clip with their best burst."""
    frames = frame_features(wave16)
    if not frames:
        return [], {}
    energy = frames["energy_db"]
    floor = float(np.percentile(energy, 2))
    speech = energy > float(np.percentile(energy, 95)) - SPEECH_BELOW_PEAK_DB
    for start, end in _runs(~speech):          # close holes shorter than 30 ms
        if end - start < 3 and start > 0 and end < len(speech):
            speech[start:end] = True
    for start, end in _runs(speech):           # drop speech islands shorter than 30 ms (clicks)
        if end - start < 3:
            speech[start:end] = False
    out = []
    count = len(speech)
    for start, end in _runs(~speech):
        kind = "lead" if start == 0 else "trail" if end == count else "gap"
        row = {"kind": kind, "start": round(start * HOP_SECONDS, 3),
               "end": round(end * HOP_SECONDS, 3), "dur": round((end - start) * HOP_SECONDS, 3)}
        row.update(best_burst(frames, start, end, floor, count))
        out.append(row)
    rms = np.sqrt(np.convolve(wave16.astype(np.float64) ** 2, np.ones(800) / 800, "valid")[::160])
    meta = {"floor_dbfs": round(float(20 * np.log10(np.percentile(rms, 2) + 1e-9)), 1),
            "zero_frac": round(float((wave16 == 0).mean()), 4)}
    return out, meta


def _to16k(wave: np.ndarray, rate: int) -> np.ndarray:
    """Integer decimation behind a windowed-sinc low-pass (7.2 kHz); numpy only."""
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if rate == SAMPLE_RATE:
        return wave.astype(np.float32)
    taps = np.arange(-120, 121)
    kernel = np.sinc(2 * 7200 / rate * taps) * np.blackman(len(taps))
    kernel /= kernel.sum()
    smooth = np.convolve(wave, kernel, "same")
    factor, remainder = divmod(rate, SAMPLE_RATE)
    if not remainder:
        return smooth[::factor].astype(np.float32)
    # Rasa mixes 48 kHz and 22.05 kHz files; linear interpolation of the band-limited
    # signal is ample for frame energies and band fractions.
    grid = np.arange(int(len(wave) * SAMPLE_RATE / rate)) * (rate / SAMPLE_RATE)
    return np.interp(grid, np.arange(len(wave)), smooth).astype(np.float32)


def scan_shard(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    import soundfile as sf
    rows, index = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        for record in batch.to_pylist():
            wave, rate = sf.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
            found, meta = segments(_to16k(wave, rate))
            for segment in found:
                rows.append({"clip": record["filename"], "speaker": VOICE[record["gender"]],
                             "style": record["style"], "shard": Path(path).name, "row": index,
                             "clip_s": round(len(wave) / rate, 3), **meta, **segment})
            index += 1
    return rows


def scan(args: argparse.Namespace) -> int:
    shards = sorted(glob.glob(RASA))[: args.shards or None]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "segments.csv").open("w", newline="", encoding="utf-8") as handle, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for done, rows in enumerate(pool.map(scan_shard, shards), 1):
            writer.writerows(rows)
            print(f"{done}/{len(shards)} shards", flush=True)
    return 0


def control(_: argparse.Namespace) -> int:
    """The burst statistic against C4's labelled gaps (telephone speech, real breaths)."""
    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measure_breath_detectability import BAND, auc
    alignments = json.loads(ALIGNMENTS.read_text(encoding="utf-8"))
    frames_len, labels, hits = [], [], []
    for record in alignments["clips"]:
        if record["label"] != "positive":
            continue
        gaps = [g for g in record["gaps"] if BAND[0] <= g["seconds"] <= BAND[1]]
        if not gaps:
            continue
        wave, _ = sf.read(str(CALIBRATION / "clips" / f"{record['id']}.flac"), dtype="float32")
        frames = frame_features(wave if wave.ndim == 1 else wave.mean(axis=1))
        if not frames:
            continue
        floor = float(np.percentile(frames["energy_db"], 2))
        count = len(frames["energy_db"])
        for gap in gaps:
            first = max(0, int(gap["start"] / HOP_SECONDS))
            last = min(count, int(gap["end"] / HOP_SECONDS) + 1)
            if last - first < 2:
                continue
            # aligner boundaries are not acoustic edges, so no isolation guard here
            burst = best_burst(frames, first, last, floor, count, False, False)
            frames_len.append(burst["burst_frames"])
            hits.append(burst["burst_frames"] >= MIN_BURST_FRAMES)
            labels.append(1 if gap["has_breath"] else 0)
    frames_len, labels, hits = np.array(frames_len, float), np.array(labels), np.array(hits)
    result = {
        "gaps": int(len(labels)), "breath_gaps": int(labels.sum()),
        "auc_burst_length": round(auc(frames_len, labels), 4),
        "hit_rate_breath_gaps": round(float(hits[labels == 1].mean()), 4),
        "hit_rate_unmarked_gaps": round(float(hits[labels == 0].mean()), 4),
        "precision_of_a_hit": round(float(labels[hits].mean()), 4) if hits.any() else None,
        "note": "burst length is correlated with gap length, which D-030 showed carries "
                "AUC 0.824 by itself; this control shows the statistic fires on real "
                "breaths, not that it is a publishable detector.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "control.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def listen(args: argparse.Namespace) -> int:
    """Top candidates per voice plus random long pauses, as original 48 kHz audio."""
    import pyarrow.parquet as pq
    import soundfile as sf
    rows = list(csv.DictReader((OUT / "segments.csv").open(encoding="utf-8")))
    for row in rows:
        for key in ("burst_frames", "isolated", "row"):
            row[key] = int(row[key])
        for key in ("burst_db", "dur", "burst_start"):
            row[key] = float(row[key])
    rng = np.random.default_rng(20261112)
    picks: list[tuple[str, dict]] = []
    for voice in ("anitha", "arun"):
        mine = [r for r in rows if r["speaker"] == voice]
        cands = sorted((r for r in mine if r["isolated"] and r["burst_frames"] >= MIN_BURST_FRAMES),
                       key=lambda r: -(r["burst_frames"] * r["burst_db"]))
        seen, top = set(), []
        for r in cands:
            if r["clip"] not in seen:
                seen.add(r["clip"])
                top.append(r)
            if len(top) == args.top:
                break
        picks += [("top", r) for r in top]
        plain = [r for r in mine if r["kind"] == "gap" and r["dur"] >= 0.25 and r["clip"] not in seen]
        for i in rng.choice(len(plain), size=min(args.random, len(plain)), replace=False):
            picks.append(("random_pause", plain[int(i)]))
    target = OUT / "listen"
    target.mkdir(parents=True, exist_ok=True)
    sheet = []
    for n, (why, r) in enumerate(picks, 1):
        table = pq.ParquetFile(str(Path(RASA).parent / r["shard"])).read().slice(r["row"], 1).to_pylist()[0]
        assert table["filename"] == r["clip"], (table["filename"], r["clip"])
        name = f"{n:02d}_{r['speaker']}_{r['style']}_{why}.wav"
        wave, rate = sf.read(io.BytesIO(table["audio"]["bytes"]), dtype="float32")
        sf.write(str(target / name), wave, rate)
        at = r["burst_start"] if r["burst_frames"] else float(r["start"])
        sheet.append(f"{name}\tlisten at {at:.2f}s of {float(r['clip_s']):.2f}s\t{r['kind']}\t"
                     f"burst {r['burst_frames'] * 10} ms, {r['burst_db']:.1f} dB over floor\t{table['text']}")
    (target / "LISTEN.tsv").write_text("\n".join(sheet), encoding="utf-8")
    print(f"wrote {len(picks)} clips to {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_studio_breaths")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--shards", type=int, default=0, help="first N shards only (0 = all)")
    s.add_argument("--workers", type=int, default=6)
    s.set_defaults(run=scan)
    sub.add_parser("control").set_defaults(run=control)
    li = sub.add_parser("listen")
    li.add_argument("--top", type=int, default=7)
    li.add_argument("--random", type=int, default=3)
    li.set_defaults(run=listen)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
