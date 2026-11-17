r"""ETL step 1: audio QC over every TTS training clip.

    py -3.11 scripts\run_audio_qc.py [--source indicvoices_r|rasa] [--limit N] [--force]

Scope, and why it is narrower than runbook section 5 step 1
-----------------------------------------------------------
Step 1 as written also covers Kathbath (m4a), Common Voice (mp3) and OpenSLR 127's
speaker IDs. All three are **ASR-only** corpora, and D-032 cut ASR from v1: Common Voice
is eval-only by D-012, and Kathbath and OpenSLR 127 appear in no TTS manifest. Decoding
them now would be several hours of work for a release that will not use them, so this
runs the TTS side only. The ASR side of step 1 is not deleted -- it is deferred with the
rest of ASR to v1.1, and this docstring is the record of that.

IndicVoices itself is also absent, deliberately. Stage A v0 (D-034) does not use its
audio: step 3b needs only its *text* columns for the mark join, and its Conversation
audio is step 3's input, which D-034 moved behind M2. Its 804 h will be QC'd when
restoration runs, not before.

Output: one parquet per input shard under ``work\audio_qc\<source>\``, merged into
``work\audio_qc.parquet``. Resumable -- re-running picks up where a dropped drive left
off.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.audio.qc_scan import ShardSpec, scan_shards  # noqa: E402

ROOT = Path(r"E:\Geetha")
WORK = ROOT / "work" / "audio_qc"
MERGED = ROOT / "work" / "audio_qc.parquet"

SOURCES = {
    "indicvoices_r": (
        ROOT / "raw" / "hf" / "indicvoices_r" / "Tamil",
        "train-*.parquet",
        ShardSpec("indicvoices_r", "audio",
                  carry=("speaker_id", "scenario", "task_name", "gender")),
    ),
    "rasa": (
        ROOT / "raw" / "hf" / "rasa" / "Tamil",
        "train-*.parquet",
        ShardSpec("rasa", "audio", carry=("filename", "style", "gender")),
    ),
}


def merge() -> int:
    """Concatenate the per-shard outputs into one table. Schemas differ by source."""
    import pyarrow as pa

    tables = []
    for name in SOURCES:
        files = sorted((WORK / name).glob("*.parquet"))
        if files:
            tables.append(pa.concat_tables([pq.read_table(f) for f in files]))
    if not tables:
        print("nothing to merge yet")
        return 0
    merged = pa.concat_tables(tables, promote_options="default")
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, MERGED)
    print(f"merged {merged.num_rows:,} rows -> {MERGED}")
    return merged.num_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), action="append")
    parser.add_argument("--limit", type=int, help="first N shards only, for a smoke run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    if args.merge_only:
        return 0 if merge() else 1

    for name in args.source or sorted(SOURCES):
        directory, pattern, spec = SOURCES[name]
        shards = sorted(directory.glob(pattern))
        if args.limit:
            shards = shards[: args.limit]
        if not shards:
            print(f"!! {name}: no shards matched {directory / pattern}", file=sys.stderr)
            return 1
        print(f"== {name}: {len(shards)} shards", flush=True)
        started = time.time()
        written = scan_shards(shards, spec, WORK / name, force=args.force, progress=True)
        print(f"== {name}: {len(written)} written, "
              f"{len(shards) - len(written)} already done, {time.time() - started:.0f}s",
              flush=True)

    merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
