r"""C3 — calibrate the breath detector against real Tamil speech.

The release headline (D-021) is that Indic-Speak's non-verbal tokens are inert, so a
Geetha that breathes is doing something the base model cannot do at all. Publishing
that as a number needs a detector, and a detector needs a threshold set where precision
and recall were actually measured — not chosen by ear.

What this run does
------------------
``dhevenddra/geetha-breath-calibration`` holds 800 real Tamil clips built by
``scripts/make_breath_calibration.py``: 400 whose IndicVoices annotation contains at
least one ``[breathing]`` or ``[sigh]`` mark, and 400 carrying **no annotation at all**.
Clips with a filler or another non-verbal event but no breath were excluded as
ambiguous, because a hesitation is breathy and scoring one as a clean negative would
charge the detector with a false positive it never made.

This run scores every clip with a public AudioSet-trained tagger and reports the
precision/recall curve over the threshold, so the operating point is a measured choice
with a cost attached rather than a preference.

Two numbers, because they answer different questions
----------------------------------------------------
* **Clip-level** — does this utterance contain an audible breath? Binary, which is what
  the headline claim is, and what AST natively produces.
* **Rate** — breaths per minute, via windowed scoring and ``geetha.audio.events``.
  Duration-normalised, so it is comparable across systems whose utterance lengths
  differ. In this set the positives run about 10 s and the negatives about 6 s, which is
  causal rather than an artefact (a longer utterance needs a breath) but does flatter
  clip-level recall, so the rate is the fairer of the two.

The ground truth to beat: the positive half of this set contains 648 annotated breaths
over 71.0 minutes, i.e. **9.1 breaths per minute of real Tamil speech**. A model far
above that is not more lifelike, it is wheezing.

The detector is deliberately not ours: using a published classifier means a reader can
re-run this. Training our own would make the headline depend on a model only we have.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16_000

# AudioSet classes that count as "a breath" for the headline. Chosen to match
# geetha.eval.breath_calibration.BREATH_MARKS, which is {breath, sigh}.
BREATH_CLASSES = ("Breathing", "Sigh", "Gasp")
# Reported alongside but NOT counted: these are other non-verbal events, useful for
# seeing what the detector confuses a breath with.
NEARBY_CLASSES = ("Throat clearing", "Cough", "Sniff", "Snoring", "Speech", "Silence")

# Windowed scoring for the rate. AST reads a 10.24 s spectrogram; a 1 s window padded
# into that is context-free but localises an event well enough to count it.
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5

_MAX_MOUNT_DEPTH = 4


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"no mounted dataset contains {list(markers)}")


def show_inputs() -> None:
    root = Path("/kaggle/input")
    print(f"/kaggle/input: {[str(p.relative_to(root)) for p in _directories(root)]}",
          flush=True)


def show_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        print("no GPU — AST on CPU would take hours. Stopping.", flush=True)
        raise SystemExit("no CUDA device")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU {name}, {total:.1f} GB", flush=True)


def require_token() -> str:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        candidate = mount / "hf_read_token.txt"
        if candidate.exists():
            token = candidate.read_text(encoding="utf-8").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                print(f"HF token from private dataset, length {len(token)}", flush=True)
                return token
    return os.environ.get("HF_TOKEN", "")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "audio")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def precision_recall(scores_pos, scores_neg, threshold):
    true_positive = int((np.asarray(scores_pos) >= threshold).sum())
    false_negative = len(scores_pos) - true_positive
    false_positive = int((np.asarray(scores_neg) >= threshold).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": round(float(threshold), 4),
        "tp": true_positive, "fp": false_positive, "fn": false_negative,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> int:
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    show_inputs()
    show_gpu()
    require_token()
    import_geetha()

    from geetha.audio.events import count_events, events_per_minute

    data_dir = find_dataset("manifest.json")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    clips = manifest["clips"]
    clips_dir = data_dir / "clips"
    print(f"{len(clips)} clips, breath marks {manifest['breath_marks']}", flush=True)

    extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(MODEL_ID).eval().cuda()
    labels = model.config.id2label
    name_to_id = {name: index for index, name in labels.items()}
    missing = [c for c in BREATH_CLASSES if c not in name_to_id]
    if missing:
        raise SystemExit(f"AudioSet classes not in this checkpoint: {missing}")
    breath_ids = [name_to_id[c] for c in BREATH_CLASSES]
    nearby_ids = {c: name_to_id[c] for c in NEARBY_CLASSES if c in name_to_id}
    print(f"breath classes {dict(zip(BREATH_CLASSES, breath_ids))}", flush=True)

    @torch.no_grad()
    def score(waveforms: list[np.ndarray]) -> np.ndarray:
        inputs = extractor(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = model(inputs.input_values.cuda()).logits
        # AudioSet is multi-label: each class is an independent sigmoid, not a softmax.
        return torch.sigmoid(logits).cpu().numpy()

    results = []
    started = time.time()
    for index, clip in enumerate(clips, 1):
        wave, rate = sf.read(str(clips_dir / f"{clip['id']}.flac"), dtype="float32")
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{clip['id']} is {rate} Hz, expected {SAMPLE_RATE}")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)

        probabilities = score([wave])[0]
        clip_score = float(max(probabilities[i] for i in breath_ids))

        # Windowed pass for the rate.
        window = int(WINDOW_SECONDS * SAMPLE_RATE)
        hop = int(HOP_SECONDS * SAMPLE_RATE)
        starts = list(range(0, max(1, len(wave) - window + 1), hop))
        windows = [wave[s:s + window] for s in starts]
        window_scores = []
        for batch_start in range(0, len(windows), 16):
            batch = windows[batch_start:batch_start + 16]
            probs = score(batch)
            for row in probs:
                window_scores.append(float(max(row[i] for i in breath_ids)))

        results.append({
            "id": clip["id"],
            "label": clip["label"],
            "seconds": clip["seconds"],
            "breaths_annotated": clip["breaths"],
            "clip_score": round(clip_score, 5),
            "window_times": [round(s / SAMPLE_RATE, 3) for s in starts],
            "window_scores": [round(v, 5) for v in window_scores],
            "nearby": {name: round(float(probabilities[i]), 4)
                       for name, i in nearby_ids.items()},
        })

        if index <= 3 or index % 50 == 0 or index == len(clips):
            rate_so_far = (time.time() - started) / index
            print(f"  {index}/{len(clips)}  {rate_so_far:.2f}s/clip, "
                  f"~{rate_so_far * (len(clips) - index) / 60:.0f} min left", flush=True)
            (OUT_DIR / "c3_partial.json").write_text(
                json.dumps(results, ensure_ascii=False), encoding="utf-8")

    positives = [r for r in results if r["label"] == "positive"]
    negatives = [r for r in results if r["label"] == "negative"]
    pos_scores = [r["clip_score"] for r in positives]
    neg_scores = [r["clip_score"] for r in negatives]

    sweep = [precision_recall(pos_scores, neg_scores, t)
             for t in np.round(np.arange(0.01, 0.99, 0.01), 2)]
    best_f1 = max(sweep, key=lambda row: row["f1"])
    # An operating point where a detection is trustworthy: the headline is about
    # Geetha producing breaths, so a false positive is worse than a missed one.
    high_precision = next(
        (row for row in sweep if row["precision"] >= 0.95 and row["recall"] > 0),
        None,
    )

    chosen = high_precision or best_f1
    threshold = chosen["threshold"]

    rate_rows = []
    for row in results:
        spans = count_events(
            list(zip(row["window_times"], row["window_scores"])),
            threshold=threshold, hop_seconds=HOP_SECONDS,
        )
        rate_rows.append({
            "id": row["id"], "label": row["label"], "seconds": row["seconds"],
            "detected": len(spans), "annotated": row["breaths_annotated"],
        })
    detected_pos = sum(r["detected"] for r in rate_rows if r["label"] == "positive")
    seconds_pos = sum(r["seconds"] for r in rate_rows if r["label"] == "positive")
    annotated_pos = sum(r["annotated"] for r in rate_rows if r["label"] == "positive")
    detected_neg = sum(r["detected"] for r in rate_rows if r["label"] == "negative")
    seconds_neg = sum(r["seconds"] for r in rate_rows if r["label"] == "negative")

    summary = {
        "model": MODEL_ID,
        "breath_classes": list(BREATH_CLASSES),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "chosen_threshold": threshold,
        "chosen_by": "precision>=0.95" if high_precision else "max F1",
        "at_threshold": chosen,
        "best_f1": best_f1,
        "sweep": sweep,
        "rate_detected_per_min_positive": round(
            events_per_minute(detected_pos, seconds_pos), 2),
        "rate_annotated_per_min_positive": round(
            events_per_minute(annotated_pos, seconds_pos), 2),
        "rate_detected_per_min_negative": round(
            events_per_minute(detected_neg, seconds_neg), 2),
        "per_clip": rate_rows,
    }
    (OUT_DIR / "c3_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "c3_scores.json").write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# C3 — breath detector calibration",
        "",
        f"Detector: `{MODEL_ID}`, AudioSet classes {list(BREATH_CLASSES)}.",
        f"{len(positives)} positive and {len(negatives)} negative real Tamil clips.",
        "",
        "## Operating point",
        "",
        f"**Threshold {threshold}** (chosen by {summary['chosen_by']}): "
        f"precision **{chosen['precision']:.3f}**, recall **{chosen['recall']:.3f}**, "
        f"F1 {chosen['f1']:.3f} — {chosen['tp']} true / {chosen['fp']} false positives, "
        f"{chosen['fn']} missed.",
        "",
        f"Best F1 anywhere on the sweep: {best_f1['f1']:.3f} at {best_f1['threshold']} "
        f"(precision {best_f1['precision']:.3f}, recall {best_f1['recall']:.3f}).",
        "",
        "A false positive is worse than a missed breath here: the claim is that Geetha",
        "produces breaths the base model cannot, so a detector that invents them would",
        "manufacture the result. Hence the high-precision operating point where one exists.",
        "",
        "## Rate",
        "",
        "| | detected/min | annotated/min |",
        "| --- | ---: | ---: |",
        f"| positive clips | {summary['rate_detected_per_min_positive']} | "
        f"{summary['rate_annotated_per_min_positive']} |",
        f"| negative clips | {summary['rate_detected_per_min_negative']} | 0.00 |",
        "",
        "The annotated rate on the positive half is the ground truth for real Tamil",
        "speech. A synthetic voice far above it is not more lifelike, it is wheezing;",
        "the negative row is the detector's own false-positive rate on real speech.",
        "",
        "## Threshold sweep",
        "",
        "| threshold | precision | recall | F1 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in sweep[::10]:
        lines.append(f"| {row['threshold']} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | {row['f1']:.3f} |")
    lines += [
        "",
        "## Caveats, stated rather than buried",
        "",
        "* Calibrated on **real** speech and applied to **synthetic** speech, which is a",
        "  different acoustic domain. Naturalness-Ta's 24 `clean` prompts give the",
        "  false-positive rate in the synthetic domain, on the systems being compared.",
        "* Positive clips are longer than negative ones (a longer utterance needs a",
        "  breath). That is causal, not an artefact of selection, but it does flatter",
        "  clip-level recall — which is why the rate above is the fairer figure.",
        "* The annotations mark a breath's position **in the text**, not a timestamp, so",
        "  this calibrates counts per clip, not onsets.",
    ]
    (OUT_DIR / "c3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
