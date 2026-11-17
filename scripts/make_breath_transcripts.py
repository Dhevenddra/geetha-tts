r"""Recover the verbatim transcript behind every clip in the breath calibration set.

    py -3.11 scripts\make_breath_transcripts.py

Why this exists
---------------
``make_breath_calibration.py`` wrote 800 clips and a manifest holding a breath **count**
per clip, and deliberately nothing else: its docstring records that the marks give "the
position in the text, not a timestamp", so a count "is all we can honestly claim from
these labels".

D-029 is the measurement that overturns the second half of that. IndicVoices places each
``[breathing]`` mark inline at the point it occurs, and 2,816 of 2,845 marks in a single
shard have words on **both sides** - so the mark names the gap between two specific words.
Align those words to the audio and the breath has a time window. That is what turns 648
weak count labels into 648 located pauses, and it needs the transcript text, which the
manifest does not carry.

Why it replays the selection instead of re-matching
---------------------------------------------------
Guessing which corpus row produced ``pos_0137`` by matching on speaker and mark counts
would be a near-miss waiting to happen, and a wrong transcript would put a breath window
in the wrong place - a silent corruption of the ground truth rather than a crash.

The original selection depends only on ``(speaker_id, text, duration, snr, marks)`` and
never on the audio bytes, so it can be **replayed exactly** while reading metadata columns
only, which is also fast. The replay is then checked clip by clip against the shipped
manifest on ``(speaker_id, snr, breaths, marks)``, and the output is written only if all
800 match. If the two ever disagree, this script fails rather than emitting a transcript
that may belong to a different utterance.

One trap worth naming: the IndicVoices-R directory holds a ``test-`` shard alongside the
``train-`` shards, and ``test-`` sorts first. Globbing ``*.parquet`` instead of
``train-*.parquet`` scans it first and shifts every clip id, which reproduces 0 of 800.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
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
from geetha.text.normalise import norm_text

RAW = Path(r"E:\Geetha\raw")
OUT_DIR = Path(r"E:\Geetha\work\breath_calibration")
HOLDOUT = Path(r"E:\Geetha\eval_vault\naturalness_ta_v1\ivr_heldout_speakers.json")

# Must match make_breath_calibration.py exactly, or the replay is not a replay.
INDICVOICES = r"hf\indicvoices\tamil\train-*.parquet"
INDICVOICES_R = r"hf\indicvoices_r\Tamil\train-*.parquet"
MAX_CLIPS_PER_SPEAKER = 3
PER_CLASS = 400


def build_verbatim_index() -> dict[str, tuple[dict[str, int], str]]:
    """``(speaker_id, norm_text)`` -> (mark counts, verbatim text)."""
    files = sorted(glob.glob(str(RAW / INDICVOICES)))
    if not files:
        raise SystemExit(f"no IndicVoices shards matched {INDICVOICES}")
    index: dict[str, tuple[dict[str, int], str]] = {}
    for path in files:
        table = pq.read_table(path, columns=["speaker_id", "text", "unsanitized_verbatim"])
        for speaker, text, verbatim in zip(table.column("speaker_id").to_pylist(),
                                           table.column("text").to_pylist(),
                                           table.column("unsanitized_verbatim").to_pylist()):
            if not speaker or not text:
                continue
            index[f"{speaker}\u0000{norm_text(text)}"] = (count_marks(verbatim), verbatim or "")
    print(f"  IndicVoices: {len(index):,} distinct keys from {len(files)} shards", flush=True)
    return index


def replay_selection(index: dict[str, tuple[dict[str, int], str]]) -> list[dict]:
    """The same scan make_breath_calibration.py performs, without decoding audio."""
    held_out = set(json.loads(HOLDOUT.read_text(encoding="utf-8"))["held_out"])
    files = sorted(glob.glob(str(RAW / INDICVOICES_R)))
    if not files:
        raise SystemExit(f"no IndicVoices-R shards matched {INDICVOICES_R}")
    print(f"  excluding {len(held_out)} holdout speakers; "
          f"scanning {len(files)} IndicVoices-R shards", flush=True)

    kept: dict[str, list[dict]] = {POSITIVE: [], NEGATIVE: []}
    per_speaker: collections.Counter = collections.Counter()
    for number, path in enumerate(files, 1):
        if all(len(kept[label]) >= PER_CLASS for label in kept):
            print(f"  both classes full after {number - 1} shards", flush=True)
            break
        table = pq.read_table(path, columns=["speaker_id", "text", "duration", "snr"])
        speakers = table.column("speaker_id").to_pylist()
        texts = table.column("text").to_pylist()
        durations = table.column("duration").to_pylist()
        snrs = table.column("snr").to_pylist()

        for row, (speaker, text) in enumerate(zip(speakers, texts)):
            if not speaker or not text or speaker in held_out:
                continue
            hit = index.get(f"{speaker}\u0000{norm_text(text)}")
            if hit is None:
                continue
            marks, verbatim = hit
            label = label_for(marks)
            if label == AMBIGUOUS or len(kept[label]) >= PER_CLASS:
                continue
            if per_speaker[(label, speaker)] >= MAX_CLIPS_PER_SPEAKER:
                continue
            if rejection_reason({"duration": durations[row], "snr": snrs[row], "marks": marks}):
                continue
            per_speaker[(label, speaker)] += 1
            kept[label].append({
                "id": f"{label[:3]}_{len(kept[label]):04d}",
                "label": label,
                "speaker_id": speaker,
                "snr": round(float(snrs[row]), 1),
                "breaths": sum(marks.get(mark, 0) for mark in BREATH_MARKS),
                "marks": marks,
                "text": text,
                "verbatim": verbatim,
            })
    return kept[POSITIVE] + kept[NEGATIVE]


def verify(replayed: list[dict], manifest_clips: list[dict]) -> tuple[int, list[str]]:
    """Exact matches, and the ids that disagree. Identity is the whole point here."""
    by_id = {row["id"]: row for row in replayed}
    exact, wrong = 0, []
    for clip in manifest_clips:
        row = by_id.get(clip["id"])
        if (row is not None
                and row["speaker_id"] == clip["speaker_id"]
                and row["snr"] == clip["snr"]
                and row["breaths"] == clip["breaths"]
                and row["marks"] == clip["marks"]):
            exact += 1
        else:
            wrong.append(clip["id"])
    return exact, wrong


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_breath_transcripts")
    parser.add_argument("--out", default=str(OUT_DIR / "transcripts.json"))
    args = parser.parse_args(argv)

    manifest_path = OUT_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"no calibration manifest at {manifest_path}")
    manifest_clips = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"]

    print("indexing IndicVoices annotations...", flush=True)
    index = build_verbatim_index()
    print("replaying the calibration-set selection...", flush=True)
    replayed = replay_selection(index)

    exact, wrong = verify(replayed, manifest_clips)
    print(f"\nreplayed {len(replayed)} clips against a manifest of {len(manifest_clips)}")
    print(f"exact match on (speaker, snr, breaths, marks): {exact}")
    if wrong:
        print(f"MISMATCHED or missing: {len(wrong)} -> {wrong[:10]}")
        print("refusing to write: a transcript that belongs to a different utterance would "
              "put a breath window in the wrong place.")
        return 1

    out = Path(args.out)
    payload = {row["id"]: {"text": row["text"], "verbatim": row["verbatim"],
                           "label": row["label"], "breaths": row["breaths"]}
               for row in replayed}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    marks = sum(row["breaths"] for row in replayed)
    print(f"wrote {out}  ({len(payload)} clips, {marks} annotated breaths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
