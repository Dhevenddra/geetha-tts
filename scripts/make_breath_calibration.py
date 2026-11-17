r"""Build the set of real Tamil clips that calibrates the breath detector.

    py -3.11 -m scripts.make_breath_calibration [--per-class 400]

Writes ``E:\Geetha\work\breath_calibration\`` — 16 kHz mono FLAC clips plus a manifest
and a report — ready to push as a private Kaggle dataset for the detector run.

Why this exists
---------------
The release headline is a rate: "Geetha N breaths/min, Indic-Speak 0.0". A rate needs a
detector, a detector needs a threshold, and a threshold chosen by taste is a number we
could not defend. IndicVoices annotates ``[breathing]`` **inline at the position it
occurs** in ``unsanitized_verbatim``; IndicVoices-R carries the restored wideband audio
for the same utterances. Joining them on ``(speaker_id, norm_text(text))`` — 97.4%,
measured as D-018 — gives real Tamil speech whose breath count is known, so the
threshold can be set where precision and recall are actually measured.

What is deliberately excluded
-----------------------------
* **The TTSDS2 holdout speakers.** They are not needed here and keeping them untouched
  means the distributional comparison stays about speakers nothing else has used.
* **Clips with any annotated noise**, persistent or transient: a detector calibrated on
  them learns to fire on noise.
* **Clips with a filler or another non-verbal event but no breath.** These are neither
  positive nor negative — see ``geetha.eval.breath_calibration``. Scoring a hesitation
  as a clean negative would charge the detector with a false positive it never made,
  push the threshold up, and then under-count our own model's breaths.

Audio is converted to 16 kHz mono because that is what the AudioSet-trained tagger
expects, and because 48 kHz stereo WAV is roughly 24x larger for no benefit.
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from geetha.eval.breath_calibration import (
    AMBIGUOUS,
    BREATH_MARKS,
    NEGATIVE,
    POSITIVE,
    count_marks,
    label_for,
    rejection_reason,
)
from geetha.runlog import RunLog
from geetha.text.normalise import norm_text

RAW = Path(r"E:\Geetha\raw")
OUT_DIR = Path(r"E:\Geetha\work\breath_calibration")
HOLDOUT = Path(r"E:\Geetha\eval_vault\naturalness_ta_v1\ivr_heldout_speakers.json")

INDICVOICES = r"hf\indicvoices\tamil\train-*.parquet"
INDICVOICES_R = r"hf\indicvoices_r\Tamil\train-*.parquet"

TARGET_SAMPLE_RATE = 16_000
# No single speaker should dominate either class: a threshold tuned to one person's
# breathing is a threshold tuned to one person.
MAX_CLIPS_PER_SPEAKER = 3
SEED = 8642


def build_mark_index(log: RunLog) -> dict[str, dict[str, int]]:
    """``(speaker_id, norm_text)`` -> annotation counts, from IndicVoices."""
    files = sorted(glob.glob(str(RAW / INDICVOICES)))
    if not files:
        raise SystemExit(f"no IndicVoices shards matched {INDICVOICES}")
    index: dict[str, dict[str, int]] = {}
    rows = 0
    for path in files:
        table = pq.read_table(path, columns=["speaker_id", "text", "unsanitized_verbatim"])
        speakers = table.column("speaker_id").to_pylist()
        texts = table.column("text").to_pylist()
        verbatims = table.column("unsanitized_verbatim").to_pylist()
        for speaker, text, verbatim in zip(speakers, texts, verbatims):
            if not speaker or not text:
                continue
            index[f"{speaker}\u0000{norm_text(text)}"] = count_marks(verbatim)
            rows += 1
    print(f"  IndicVoices: {rows:,} rows -> {len(index):,} distinct keys "
          f"from {len(files)} shards", flush=True)
    log.add_param("indicvoices_rows", rows)
    log.add_param("indicvoices_keys", len(index))
    return index


def to_flac(wav_bytes: bytes, destination: Path) -> float | None:
    """Convert to 16 kHz mono FLAC with ffmpeg. Returns duration, or None on failure."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", "pipe:0", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
         "-c:a", "flac", str(destination)],
        input=wav_bytes, capture_output=True,
    )
    if result.returncode != 0 or not destination.exists():
        return None
    import soundfile as sf

    info = sf.info(str(destination))
    return info.frames / info.samplerate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_breath_calibration")
    parser.add_argument("--per-class", type=int, default=400,
                        help="clips to keep per class (default 400)")
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips_dir = OUT_DIR / "clips"
    clips_dir.mkdir(exist_ok=True)

    log = RunLog(run_id="b-breath-calibration", seed=SEED, out_dir=OUT_DIR)
    log.add_param("per_class", args.per_class)
    log.add_param("sample_rate", TARGET_SAMPLE_RATE)
    log.add_param("breath_marks", sorted(BREATH_MARKS))

    held_out = set(json.loads(HOLDOUT.read_text(encoding="utf-8"))["held_out"])
    print(f"excluding {len(held_out)} TTSDS2 holdout speakers", flush=True)

    print("indexing IndicVoices annotations...", flush=True)
    index = build_mark_index(log)

    files = sorted(glob.glob(str(RAW / INDICVOICES_R)))
    if not files:
        raise SystemExit(f"no IndicVoices-R shards matched {INDICVOICES_R}")
    print(f"scanning {len(files)} IndicVoices-R shards...", flush=True)

    kept: dict[str, list[dict]] = {POSITIVE: [], NEGATIVE: []}
    per_speaker: collections.Counter[tuple[str, str]] = collections.Counter()
    rejected: collections.Counter[str] = collections.Counter()
    unmatched = matched = 0
    labels = collections.Counter()

    for shard_number, path in enumerate(files, 1):
        if all(len(kept[label]) >= args.per_class for label in kept):
            print(f"  both classes full after {shard_number - 1} shards", flush=True)
            break
        table = pq.read_table(
            path, columns=["speaker_id", "text", "audio", "duration", "snr"]
        )
        speakers = table.column("speaker_id").to_pylist()
        texts = table.column("text").to_pylist()
        durations = table.column("duration").to_pylist()
        snrs = table.column("snr").to_pylist()

        for row_number, (speaker, text) in enumerate(zip(speakers, texts)):
            if not speaker or not text or speaker in held_out:
                continue
            marks = index.get(f"{speaker}\u0000{norm_text(text)}")
            if marks is None:
                unmatched += 1
                continue
            matched += 1

            label = label_for(marks)
            labels[label] += 1
            if label == AMBIGUOUS or len(kept[label]) >= args.per_class:
                continue
            if per_speaker[(label, speaker)] >= MAX_CLIPS_PER_SPEAKER:
                continue

            reason = rejection_reason({
                "duration": durations[row_number],
                "snr": snrs[row_number],
                "marks": marks,
            })
            if reason:
                rejected[reason] += 1
                continue

            audio = table.column("audio")[row_number].as_py()
            raw_bytes = audio["bytes"] if isinstance(audio, dict) else audio
            clip_id = f"{label[:3]}_{len(kept[label]):04d}"
            seconds = to_flac(raw_bytes, clips_dir / f"{clip_id}.flac")
            if seconds is None:
                rejected["ffmpeg failed"] += 1
                continue

            per_speaker[(label, speaker)] += 1
            kept[label].append({
                "id": clip_id,
                "label": label,
                "speaker_id": speaker,
                "seconds": round(seconds, 3),
                "breaths": sum(marks.get(m, 0) for m in BREATH_MARKS),
                "marks": marks,
                "snr": round(float(snrs[row_number]), 1),
            })

        if shard_number % 10 == 0 or shard_number == len(files):
            print(f"  shard {shard_number}/{len(files)}: "
                  f"{len(kept[POSITIVE])} positive, {len(kept[NEGATIVE])} negative",
                  flush=True)

    manifest = kept[POSITIVE] + kept[NEGATIVE]
    if not manifest:
        print("no clips selected — nothing to calibrate on", file=sys.stderr)
        return 1

    (OUT_DIR / "manifest.json").write_text(
        json.dumps({
            "created": "2026-10-28",
            "sample_rate": TARGET_SAMPLE_RATE,
            "breath_marks": sorted(BREATH_MARKS),
            "max_clips_per_speaker": MAX_CLIPS_PER_SPEAKER,
            "clips": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = _report(kept, labels, rejected, matched, unmatched, per_speaker, held_out)
    (OUT_DIR / "breath_calibration_report.md").write_text(report, encoding="utf-8")

    log.add_output(OUT_DIR / "manifest.json")
    log.add_output(OUT_DIR / "breath_calibration_report.md")
    log.add_param("positives", len(kept[POSITIVE]))
    log.add_param("negatives", len(kept[NEGATIVE]))
    log.finish(status="ok")

    print(report)
    return 0


def _report(kept, labels, rejected, matched, unmatched, per_speaker, held_out) -> str:
    positives, negatives = kept[POSITIVE], kept[NEGATIVE]
    total_breaths = sum(c["breaths"] for c in positives)
    pos_seconds = sum(c["seconds"] for c in positives)
    neg_seconds = sum(c["seconds"] for c in negatives)
    speakers = {label_speaker[1] for label_speaker in per_speaker}

    lines = [
        "# Breath detector calibration set",
        "",
        "Built by `scripts/make_breath_calibration.py`. Real Tamil speech whose breath",
        "count is known, so the detector's threshold is measured rather than chosen.",
        "",
        "## What is in it",
        "",
        "| | clips | speakers | audio | breaths annotated |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| positive (>=1 breath) | {len(positives)} | "
        f"{len({c['speaker_id'] for c in positives})} | {pos_seconds / 60:.1f} min | "
        f"{total_breaths} |",
        f"| negative (no annotation at all) | {len(negatives)} | "
        f"{len({c['speaker_id'] for c in negatives})} | {neg_seconds / 60:.1f} min | 0 |",
        "",
        f"{len(speakers)} distinct speakers, at most {MAX_CLIPS_PER_SPEAKER} clips each: a",
        "threshold tuned to one person's breathing is a threshold tuned to one person.",
        "",
        f"Ground-truth rate in the positive set: "
        f"**{60 * total_breaths / max(pos_seconds, 1e-9):.1f} breaths/min**. That is what a",
        "calibrated detector should recover on this audio, and it is the figure our own",
        "model's rate will be read against.",
        "",
        "## The join",
        "",
        f"- {matched:,} IndicVoices-R rows matched an IndicVoices annotation on",
        "  `(speaker_id, norm_text(text))`",
        f"- {unmatched:,} did not match and were skipped",
        f"- {len(held_out)} TTSDS2 holdout speakers were excluded before any of this, so the",
        "  distributional comparison stays about speakers nothing else has touched",
        "",
        "## Labels seen while scanning",
        "",
        "| label | rows |",
        "| --- | ---: |",
    ]
    for label, count in labels.most_common():
        lines.append(f"| {label} | {count:,} |")
    lines += [
        "",
        "`ambiguous` is the important one: a clip carrying a filler or another non-verbal",
        "event but **no** breath. It is neither positive nor negative. Scoring a hesitation",
        "as a clean negative would charge the detector with a false positive it never made,",
        "push the calibrated threshold up, and then under-count our own model's breaths.",
        "",
        "## Clips rejected before selection",
        "",
        "| reason | clips |",
        "| --- | ---: |",
    ]
    for reason, count in rejected.most_common():
        lines.append(f"| {reason} | {count:,} |")
    lines += [
        "",
        "## What this cannot do",
        "",
        "The annotations give the position of a breath **in the text**, not a timestamp, so",
        "this calibrates a *count per clip*, not an onset detector. That is what the headline",
        "needs (events per minute) and it is all these labels honestly support.",
        "",
        "The threshold is also set on *real* speech and then applied to *synthetic* speech,",
        "which is a different acoustic domain. That is why Naturalness-Ta carries 24 `clean`",
        "prompts: they give a false-positive rate measured in the synthetic domain, on the",
        "systems actually being compared.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
