r"""ETL step 2: normalise training text and honour the leak rules.

    py -3.11 scripts\run_text_norm.py [--report-only]

Reads every held-out set, builds one index of their normalised sentences, then walks the
TTS training shards and writes a keep/drop decision with a reason for every row.

Outputs
    processed\text_norm\<source>\*.parquet   one row per training clip, with its
                                            normalisation, keep flag and drop reason
    manifests\leak_removed.csv              what was dropped, by reason

Rule 3 (runbook section 10) removes a held-out sentence from every training manifest.
The IV-R speaker holdout is applied here too, so a manifest built from these files
cannot accidentally train on an eval speaker: there is one place that decision is made,
and this is it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.data.holdout import HoldoutIndex, decide  # noqa: E402
from geetha.text.normalise import norm_text  # noqa: E402

ROOT = Path(r"E:\Geetha")
RAW = ROOT / "raw" / "hf"
VAULT = ROOT / "eval_vault"
OUT = ROOT / "processed" / "text_norm"
LEAK_CSV = ROOT / "manifests" / "leak_removed.csv"

TAMIL_LANGS = {"ta", "ta-in", "tamil"}

# Held-out text. Every one of these is either a frozen eval set or an authored prompt
# set; none may appear in training.
HOLDOUT_PARQUET = [
    ("ivr_test", RAW / "indicvoices_r" / "Tamil", "test-*.parquet", "text"),
    ("rasa_test", RAW / "rasa" / "Tamil", "test-*.parquet", "text"),
    ("indicvoices_valid", RAW / "indicvoices" / "tamil", "valid-*.parquet", "text"),
    ("sarvam_tts_general", VAULT / "sarvam_tts_general_benchmark" / "data", "*.parquet", "text"),
    ("sarvam_tts_robustness", VAULT / "sarvam_tts_robustness_benchmark" / "data", "*.parquet", "text"),
    ("sarvam_contextual_asr", VAULT / "sarvam_contextual_asr" / "ta-IN", "*.parquet", "text"),
]
HOLDOUT_JSONL = [
    ("directability_ta", VAULT / "directability_ta_v1.jsonl", ("text",)),
    ("capability_ta", VAULT / "naturalness_ta_v1" / "capability_ta_v1.jsonl", ("text", "spoken_text")),
]

# TTS training sources. ASR corpora are absent by D-032.
TRAINING = [
    ("indicvoices_r", RAW / "indicvoices_r" / "Tamil", "train-*.parquet", "text", "speaker_id"),
    ("rasa", RAW / "rasa" / "Tamil", "train-*.parquet", "text", "filename"),
]

SCHEMA = pa.schema([
    pa.field("source", pa.string()),
    pa.field("shard", pa.string()),
    pa.field("row_index", pa.int32()),
    pa.field("speaker_id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("norm_text", pa.string()),
    pa.field("keep", pa.bool_()),
    pa.field("reason", pa.string()),
])


def holdout_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    per_set: Counter[str] = Counter()

    for name, directory, pattern, column in HOLDOUT_PARQUET:
        files = sorted(directory.glob(pattern))
        if not files:
            print(f"!! {name}: nothing matched {directory / pattern}", file=sys.stderr)
            continue
        for path in files:
            available = set(pq.ParquetFile(path).schema_arrow.names)
            columns = [column] + (["language"] if "language" in available else [])
            table = pq.read_table(path, columns=columns)
            texts = table[column].to_pylist()
            if "language" in available:
                langs = [str(value or "").strip().lower() for value in table["language"].to_pylist()]
                texts = [t for t, lang in zip(texts, langs) if lang in TAMIL_LANGS]
            for text in texts:
                if text:
                    entries.append((name, text))
                    per_set[name] += 1

    for name, path, fields in HOLDOUT_JSONL:
        if not path.exists():
            print(f"!! {name}: missing {path}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for field in fields:
                if row.get(field):
                    entries.append((name, row[field]))
                    per_set[name] += 1

    print("held-out sentences read:")
    for name, count in sorted(per_set.items()):
        print(f"    {name:24s} {count:>7,}")
    return entries


def speaker_holdout() -> tuple[set[str], set[str]]:
    path = VAULT / "naturalness_ta_v1" / "ivr_heldout_speakers.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    return set(spec["held_out"]), set(spec.get("excluded_too_small", []))


def rasa_speaker(filename: object) -> str | None:
    """Rasa has two speakers, encoded in the filename as TAM_F_CONV_01105."""
    parts = str(filename or "").split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true",
                        help="count the decisions without writing processed/")
    args = parser.parse_args()

    index = HoldoutIndex(holdout_entries())
    print(f"holdout index: {len(index):,} distinct normalised sentences")
    held_out, too_small = speaker_holdout()
    print(f"IV-R speaker holdout: {len(held_out)} held out, {len(too_small)} too small")

    totals: Counter[str] = Counter()
    for source, directory, pattern, text_column, speaker_column in TRAINING:
        shards = sorted(directory.glob(pattern))
        out_dir = OUT / source
        if not args.report_only:
            out_dir.mkdir(parents=True, exist_ok=True)
        kept = dropped = 0
        reasons: Counter[str] = Counter()

        for path in shards:
            table = pq.read_table(path, columns=[text_column, speaker_column])
            texts = table[text_column].to_pylist()
            speakers = table[speaker_column].to_pylist()
            rows = []
            for position, (text, raw_speaker) in enumerate(zip(texts, speakers)):
                speaker = rasa_speaker(raw_speaker) if source == "rasa" else raw_speaker
                verdict = decide(text or "", speaker, index=index,
                                 held_out_speakers=held_out, small_speakers=too_small)
                rows.append({
                    "source": source, "shard": path.name, "row_index": position,
                    "speaker_id": None if speaker is None else str(speaker),
                    "text": text, "norm_text": norm_text(text or ""),
                    "keep": verdict.keep, "reason": verdict.reason,
                })
                if verdict.keep:
                    kept += 1
                else:
                    dropped += 1
                    reasons[verdict.reason] += 1
            if not args.report_only:
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out_dir / path.name)

        share = dropped / max(1, kept + dropped)
        print(f"\n== {source}: {kept:,} kept, {dropped:,} dropped ({share:.2%})")
        for reason, count in reasons.most_common():
            print(f"    {reason:32s} {count:>7,}")
        for reason, count in reasons.items():
            totals[f"{source}/{reason}"] += count
        totals[f"{source}/_kept"] = kept

    if not args.report_only:
        LEAK_CSV.parent.mkdir(parents=True, exist_ok=True)
        with LEAK_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "reason", "rows"])
            for key, count in sorted(totals.items()):
                source, reason = key.split("/", 1)
                writer.writerow([source, reason, count])
        print(f"\nwrote {LEAK_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
