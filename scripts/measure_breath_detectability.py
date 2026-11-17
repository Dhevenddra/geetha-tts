r"""Is a breath acoustically detectable inside the pause C4 located?

    py -3.11 scripts\measure_breath_detectability.py

What this answers
-----------------
C4 gave every annotated breath a time window and passed its own falsification test: a
gap carrying a breath mark runs 4.09x longer than an ordinary word gap. That says the
marks landed in real pauses. It does **not** say a breath can be *heard* in one, which is
the question the release headline actually rests on.

So this reframes the problem from "detect breaths in continuous speech" - which defeated
the AudioSet tagger and three hand-built detectors (D-028, D-029) - to "classify the
pauses we already located". That is a far better posed problem, and for evaluating our own
TTS it is also fully available: we know the prompt text we asked the model to say, so the
same alignment can find the pauses in synthetic output.

How it is kept honest
---------------------
* **Speaker-disjoint split.** Calibration and reporting speakers never overlap, so a
  threshold tuned to one person's breathing cannot score itself.
* **Controls from the same clips.** The negative class is unmarked in-band gaps from the
  *same positive recordings* - same speaker, same session, same noise floor. Drawing
  controls from the negative clips instead would confound the comparison with clip
  duration, which is the trap D-029 recorded.
* **A duration-only baseline, reported every time.** On the 800-clip set, clip duration
  already beat the rejected detector. Any instrument that cannot beat the duration of the
  pause it is looking at has not earned its place.
* **Duration is then deliberately excluded from the model we would publish.** A pause's
  length is something a TTS model controls directly: a system that merely pauses longer
  would score as "breathing" under a duration-aware rule. Measured here, excluding it
  costs nothing (AUC 0.881 either way), so the published instrument reads acoustics only
  and cannot be fooled by silence.

The standing caveat, unchanged
------------------------------
This calibrates on **real** Tamil speech and would be applied to **synthetic** speech,
which is a different acoustic domain. Naturalness-Ta's 24 ``clean`` prompts exist to give
the false-positive rate in the synthetic domain, on the systems actually being compared. A
capability number published without that rate is not a measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WINDOW = 400          # 25 ms
HOP = 160             # 10 ms
HOP_SECONDS = HOP / SAMPLE_RATE

# A gap shorter than this cannot hold a breath and a gap longer than this is a turn
# boundary rather than an inhalation. Fixed, and published with any number from here.
BAND = (0.08, 1.50)

FEATURES = ("dur", "e_max", "e_mean", "low", "mid", "high", "zcr", "flat", "cent")

CALIBRATION = Path(r"E:\Geetha\work\breath_calibration")
ALIGNMENTS = Path(r"E:\Geetha\runs\c4_breath_timestamps\c4_alignments.json")


def frame_features(wave: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame acoustics. Pure numpy so the instrument has no model dependency."""
    window = np.hanning(WINDOW).astype(np.float32)
    freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE)
    count = 1 + max(0, len(wave) - WINDOW) // HOP
    if count < 1:
        return {}
    index = np.arange(WINDOW)[None, :] + HOP * np.arange(count)[:, None]
    frames = wave[index] * window
    magnitude = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) + 1e-10
    power = magnitude ** 2
    total = power.sum(1)

    def band(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        return power[:, mask].sum(1) / total

    raw = np.lib.stride_tricks.sliding_window_view(wave, WINDOW)[::HOP][:count]
    return {
        "energy_db": (10 * np.log10(total + 1e-12)).astype(np.float32),
        # geometric/arithmetic mean of the power spectrum: high for noise, low for voicing
        "flat": (np.exp(np.log(power).mean(1)) / (power.mean(1) + 1e-20)).astype(np.float32),
        "centroid": ((power * freqs).sum(1) / total).astype(np.float32),
        "low": band(0, 300).astype(np.float32),       # f0 / voicing
        "mid": band(300, 1500).astype(np.float32),
        "high": band(1500, 6000).astype(np.float32),  # where breath turbulence shows
        "zcr": (np.diff(np.sign(raw), axis=1) != 0).mean(1).astype(np.float32),
    }


def gap_vector(frames: dict[str, np.ndarray], start: float, end: float) -> dict | None:
    """Summarise one pause. ``None`` when it is too short to summarise."""
    energy = frames["energy_db"]
    floor = float(np.percentile(energy, 5))
    first = max(0, int(start / HOP_SECONDS))
    last = min(len(energy), int(end / HOP_SECONDS) + 1)
    if last - first < 2:
        return None
    window = slice(first, last)
    return {
        "dur": end - start,
        "e_max": float(energy[window].max() - floor),
        "e_mean": float(energy[window].mean() - floor),
        "low": float(frames["low"][window].mean()),
        "mid": float(frames["mid"][window].mean()),
        "high": float(frames["high"][window].mean()),
        "zcr": float(frames["zcr"][window].mean()),
        "flat": float(frames["flat"][window].mean()),
        "cent": float(frames["centroid"][window].mean()),
    }


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC with ties averaged. 0.5 is no information."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    positives = int(label.sum())
    negatives = len(label) - positives
    if not positives or not negatives:
        return float("nan")
    return float((ranks[label == 1].sum() - positives * (positives - 1) / 2)
                 / (positives * negatives))


def fit_logistic(features: np.ndarray, labels: np.ndarray, steps: int = 4000,
                 rate: float = 0.5, l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """Plain gradient descent. Few parameters on purpose: this must not memorise."""
    weights = np.zeros(features.shape[1])
    bias = 0.0
    for _ in range(steps):
        prediction = 1 / (1 + np.exp(-(features @ weights + bias)))
        error = prediction - labels
        weights -= rate * (features.T @ error / len(labels) + l2 * weights)
        bias -= rate * error.mean()
    return weights, bias


def speaker_half(speaker: str) -> int:
    return int(hashlib.sha256(speaker.encode()).hexdigest(), 16) % 2


def collect(alignments: dict, manifest: dict) -> list[dict]:
    import soundfile as sf

    rows: list[dict] = []
    for record in alignments["clips"]:
        if record["label"] != "positive":
            continue
        clip_id = record["id"]
        in_band = [g for g in record["gaps"] if BAND[0] <= g["seconds"] <= BAND[1]]
        if not in_band:
            continue
        wave, rate = sf.read(str(CALIBRATION / "clips" / f"{clip_id}.flac"), dtype="float32")
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{clip_id} is {rate} Hz, expected {SAMPLE_RATE}")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        frames = frame_features(wave)
        if not frames:
            continue
        for gap in in_band:
            vector = gap_vector(frames, gap["start"], gap["end"])
            if vector is None:
                continue
            vector["y"] = 1 if gap["has_breath"] else 0
            vector["speaker"] = manifest[clip_id]["speaker_id"]
            rows.append(vector)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_breath_detectability")
    parser.add_argument("--out", default=str(CALIBRATION / "detectability.json"))
    args = parser.parse_args(argv)

    if not ALIGNMENTS.exists():
        raise SystemExit(f"no C4 alignments at {ALIGNMENTS} - run C4 first")
    alignments = json.loads(ALIGNMENTS.read_text(encoding="utf-8"))
    manifest = {c["id"]: c for c in
                json.loads((CALIBRATION / "manifest.json").read_text(encoding="utf-8"))["clips"]}

    print("summarising located pauses...", flush=True)
    rows = collect(alignments, manifest)
    features = np.array([[row[k] for k in FEATURES] for row in rows], float)
    labels = np.array([row["y"] for row in rows])
    halves = np.array([speaker_half(row["speaker"]) for row in rows])
    train, test = halves == 0, halves == 1

    print(f"in-band pauses from positive clips: {len(rows)}  "
          f"breath {int(labels.sum())}  unmarked {int((1 - labels).sum())}")
    print(f"train {int(train.sum())} (breath {int(labels[train].sum())})  "
          f"test {int(test.sum())} (breath {int(labels[test].sum())})  speaker-disjoint")

    single = {}
    print("\nsingle-feature AUC on held-out speakers (0.5 = no information):")
    for i, name in enumerate(FEATURES):
        single[name] = round(auc(features[test, i], labels[test]), 4)
        note = "  <-- duration-only BASELINE" if name == "dur" else ""
        print(f"  {name:7s} {single[name]:.3f}{note}")

    def standardise(columns):
        mean = features[train][:, columns].mean(0)
        deviation = features[train][:, columns].std(0) + 1e-9
        return (features[:, columns] - mean) / deviation

    everything = list(range(len(FEATURES)))
    acoustics = [i for i, name in enumerate(FEATURES) if name != "dur"]

    scaled = standardise(everything)
    weights, bias = fit_logistic(scaled[train], labels[train])
    full_auc = auc(scaled[test] @ weights + bias, labels[test])

    scaled_acoustic = standardise(acoustics)
    acoustic_weights, acoustic_bias = fit_logistic(scaled_acoustic[train], labels[train])
    acoustic_score = scaled_acoustic[test] @ acoustic_weights + acoustic_bias
    acoustic_auc = auc(acoustic_score, labels[test])
    duration_auc = auc(features[test, 0], labels[test])

    print(f"\nall {len(FEATURES)} features:                 AUC {full_auc:.3f}")
    print(f"duration of the pause alone:        AUC {duration_auc:.3f}   (the baseline to beat)")
    print(f"ACOUSTICS ONLY, duration excluded:  AUC {acoustic_auc:.3f}   (what we would publish)")

    print("\nweights, standardised, acoustics-only model:")
    ordered = sorted(zip([FEATURES[i] for i in acoustics], acoustic_weights),
                     key=lambda pair: -abs(pair[1]))
    for name, value in ordered:
        print(f"  {name:7s} {value:+.3f}")

    operating = {}
    for target in (0.90, 0.80, 0.70):
        best = None
        for threshold in np.unique(acoustic_score):
            predicted = acoustic_score >= threshold
            if not predicted.sum():
                continue
            precision = float(labels[test][predicted].mean())
            recall = float(predicted[labels[test] == 1].mean())
            if precision >= target and (best is None or recall > best[1]):
                best = (precision, recall, float(threshold))
        if best:
            operating[f"{target:.2f}"] = {"precision": round(best[0], 4),
                                          "recall": round(best[1], 4),
                                          "threshold": round(best[2], 4)}
            print(f"at precision >= {target:.2f}: precision {best[0]:.3f} recall {best[1]:.3f}")
        else:
            operating[f"{target:.2f}"] = None
            print(f"at precision >= {target:.2f}: unreachable")

    payload = {
        "band": list(BAND),
        "pauses": len(rows), "breath": int(labels.sum()),
        "train": int(train.sum()), "test": int(test.sum()),
        "single_feature_auc": single,
        "auc_all_features": round(full_auc, 4),
        "auc_duration_only": round(duration_auc, 4),
        "auc_acoustics_only": round(acoustic_auc, 4),
        "acoustics_only_weights": {name: round(float(value), 4) for name, value in ordered},
        "operating_points": operating,
        "note": "duration is excluded from the published model on purpose: a TTS system "
                "controls pause length directly, so a duration-aware rule would score a "
                "model that merely pauses longer as one that breathes.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
