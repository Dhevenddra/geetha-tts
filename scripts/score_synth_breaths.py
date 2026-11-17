r"""Score the pauses of synthetic clips with the breath instrument (D-030).

The pause-noise "lift" figure written by the Kaggle kernels is a ranker and is inflated
whenever a clip's floor is digital silence (-80 dB and below). This applies the real
instrument - ``geetha.audio.breath.BreathDetector``, fitted on C4's located pauses,
speaker-disjoint - to energy-located pauses in generated audio. The limits are in
``geetha.eval.breath_score``; the no-tag arm of the same voice is the control.

The fitted detector is saved once to ``work\breath_calibration\detector_v1.json`` and
re-loaded afterwards, so every later number comes from the same weights.

Usage:  python scripts\score_synth_breaths.py <dir of wavs> [--out report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geetha.audio.breath import (ACOUSTIC_FEATURES, BAND, BreathDetector,  # noqa: E402
                                 fit_detector, frame_features, gap_features)
from geetha.eval.breath_score import (AUDIBLE_ABOVE_SPEECH_DB,  # noqa: E402
                                      PAUSE_BELOW_SPEECH_DB, resample_to_instrument,
                                      score_wave)

CALIBRATION = Path(r"E:\Geetha\work\breath_calibration")
ALIGNMENTS = Path(r"E:\Geetha\runs\c4_breath_timestamps\c4_alignments.json")
DETECTOR = CALIBRATION / "detector_v1.json"
TARGET_PRECISION = 0.80


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wave, rate = sf.read(str(path), dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return wave, rate


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(len(score))
    pos = int(label.sum())
    neg = len(label) - pos
    return float((ranks[label == 1].sum() - pos * (pos - 1) / 2) / (pos * neg))


def load_or_fit_detector() -> tuple[BreathDetector, float, dict]:
    """The detector, its score threshold at 0.80 precision, and how it was measured."""
    if DETECTOR.exists():
        saved = json.loads(DETECTOR.read_text(encoding="utf-8"))
        return BreathDetector.from_dict(saved["detector"]), saved["threshold"], saved["held_out"]

    alignments = json.loads(ALIGNMENTS.read_text(encoding="utf-8"))
    manifest = {c["id"]: c for c in
                json.loads((CALIBRATION / "manifest.json").read_text(encoding="utf-8"))["clips"]}
    rows = []
    for record in alignments["clips"]:
        if record["label"] != "positive":
            continue
        in_band = [g for g in record["gaps"] if BAND[0] <= g["seconds"] <= BAND[1]]
        if not in_band:
            continue
        wave, rate = _read(CALIBRATION / "clips" / f"{record['id']}.flac")
        frames = frame_features(resample_to_instrument(wave, rate))
        for gap in in_band:
            vector = gap_features(frames, gap["start"], gap["end"])
            if vector is None:
                continue
            speaker = manifest[record["id"]]["speaker_id"]
            half = int(hashlib.sha256(speaker.encode()).hexdigest(), 16) % 2
            rows.append((vector, 1 if gap["has_breath"] else 0, half))
    features = np.array([[r[0][k] for k in ACOUSTIC_FEATURES] for r in rows])
    labels = np.array([r[1] for r in rows])
    train = np.array([r[2] for r in rows]) == 0
    detector = fit_detector(features[train], labels[train], ACOUSTIC_FEATURES)
    scores = np.array([detector.score(r[0]) for r in rows])
    test_scores, test_labels = scores[~train], labels[~train]
    threshold, at = 1.0, {}
    for candidate in np.sort(np.unique(test_scores)):
        picked = test_scores >= candidate
        precision = float(test_labels[picked].mean())
        if precision >= TARGET_PRECISION:
            threshold = float(candidate)
            at = {"precision": round(precision, 4),
                  "recall": round(float(picked[test_labels == 1].mean()), 4)}
            break
    held_out = {"auc": round(_auc(test_scores, test_labels), 4), "n_test": int((~train).sum()),
                "n_train": int(train.sum()), "operating_point": at}
    DETECTOR.write_text(json.dumps({"detector": detector.to_dict(), "threshold": threshold,
                                    "held_out": held_out}, indent=2), encoding="utf-8")
    return detector, threshold, held_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="score_synth_breaths")
    parser.add_argument("folder")
    parser.add_argument("--out")
    parser.add_argument("--glob", default="*.wav")
    args = parser.parse_args(argv)

    detector, threshold, held_out = load_or_fit_detector()
    print(f"detector: held-out AUC {held_out['auc']} on {held_out['n_test']} pauses; "
          f"threshold {threshold:.3f} = {held_out['operating_point']}")
    clips = []
    for path in sorted(Path(args.folder).glob(args.glob)):
        clip = {"file": path.name, **score_wave(*_read(path), detector, threshold)}
        clips.append(clip)
        hits = [(p["at_s"], p["dur_s"], p["where"], p["score"], p["level_db_re_speech"])
                for p in clip["pauses"] if p["breath"]]
        print(f"{clip['file']:<44} pauses {len(clip['pauses']):>2}  breaths {clip['breaths']}"
              f"  max {clip['max_score']}  {hits}")
    report = {"detector": held_out, "threshold": threshold,
              "pause_rule_db_below_speech": PAUSE_BELOW_SPEECH_DB,
              "audible_floor_db_re_speech": AUDIBLE_ABOVE_SPEECH_DB,
              "limits": "pauses located by energy, not alignment; synthetic-domain "
                        "false-positive rate not established - compare against the no-tag arm",
              "clips": clips}
    out = Path(args.out) if args.out else Path(args.folder) / "breath_scores.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
