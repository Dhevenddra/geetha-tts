r"""ETL step 3b: recover IndicVoices event marks for IV-R rows (D-018).

    py -3.11 scripts\run_mark_join.py

Builds the ``(speaker_id, norm_text)`` index from IndicVoices' ``unsanitized_verbatim``
-- text columns only, so no audio is decoded -- then attaches the marked text to every
IV-R training row step 2 kept.

Outputs
    processed\ivr_marks\*.parquet       one row per IV-R training clip: the marked text
                                        where a twin was found, and the event counts
    processed\ivr_marks\report.md       join rate and recovered marks per task_name

The runbook records 97.4% of IV-R rows joining and 76.0% carrying marks, measured
26 Oct. Those are re-measured here rather than quoted: the runbook is a hypothesis, and
this run is on the leak-filtered row set, which is not the set that figure came from.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.data.mark_join import MarkIndex, count_marks  # noqa: E402

ROOT = Path(r"E:\Geetha")
INDICVOICES = ROOT / "raw" / "hf" / "indicvoices" / "tamil"
IVR = ROOT / "raw" / "hf" / "indicvoices_r" / "Tamil"
TEXT_NORM = ROOT / "processed" / "text_norm" / "indicvoices_r"
OUT = ROOT / "processed" / "ivr_marks"

SCHEMA = pa.schema([
    pa.field("shard", pa.string()),
    pa.field("row_index", pa.int32()),
    pa.field("speaker_id", pa.string()),
    pa.field("task_name", pa.string()),
    pa.field("scenario", pa.string()),
    pa.field("duration", pa.float64()),
    pa.field("text", pa.string()),
    pa.field("marked_text", pa.string()),
    pa.field("joined", pa.bool_()),
    pa.field("n_breath", pa.int32()),
    pa.field("n_events", pa.int32()),
])


def build_index() -> MarkIndex:
    shards = sorted(INDICVOICES.glob("train-*.parquet"))
    print(f"reading {len(shards)} IndicVoices shards for marks (text columns only)")
    entries: list[tuple[str, str, str | None]] = []
    for position, path in enumerate(shards, start=1):
        table = pq.read_table(path, columns=["speaker_id", "text", "unsanitized_verbatim"])
        entries.extend(zip(table["speaker_id"].to_pylist(),
                           table["text"].to_pylist(),
                           table["unsanitized_verbatim"].to_pylist()))
        if position % 30 == 0:
            print(f"    {position}/{len(shards)} shards, {len(entries):,} rows", flush=True)
    index = MarkIndex(entries)
    print(f"index: {len(index):,} distinct (speaker, sentence) keys from {len(entries):,} rows; "
          f"{index.ambiguous:,} ambiguous keys")
    return index


def main() -> int:
    index = build_index()
    OUT.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    per_task: dict[str, Counter] = defaultdict(Counter)

    shards = sorted(IVR.glob("train-*.parquet"))
    for position, path in enumerate(shards, start=1):
        keep_table = pq.read_table(TEXT_NORM / path.name, columns=["row_index", "keep"])
        keep = set(index_ for index_, flag in zip(keep_table["row_index"].to_pylist(),
                                                  keep_table["keep"].to_pylist()) if flag)
        table = pq.read_table(path, columns=["speaker_id", "text", "task_name", "scenario", "duration"])
        speakers = table["speaker_id"].to_pylist()
        texts = table["text"].to_pylist()
        tasks = table["task_name"].to_pylist()
        scenarios = table["scenario"].to_pylist()
        durations = table["duration"].to_pylist()

        rows = []
        for row_index in range(table.num_rows):
            if row_index not in keep:
                continue
            marked = index.lookup(speakers[row_index], texts[row_index] or "")
            counts = count_marks(marked)
            breaths = counts.get("<|nv_breath|>", 0)
            events = sum(counts.values())
            task = tasks[row_index] or "(none)"
            rows.append({
                "shard": path.name, "row_index": row_index,
                "speaker_id": speakers[row_index], "task_name": task,
                "scenario": scenarios[row_index],
                "duration": durations[row_index],
                "text": texts[row_index], "marked_text": marked,
                "joined": marked is not None,
                "n_breath": breaths, "n_events": events,
            })
            totals["rows"] += 1
            totals["joined"] += int(marked is not None)
            totals["with_events"] += int(events > 0)
            totals["breaths"] += breaths
            totals["events"] += events
            totals["seconds"] += durations[row_index] or 0.0
            bucket = per_task[task]
            bucket["rows"] += 1
            bucket["joined"] += int(marked is not None)
            bucket["with_events"] += int(events > 0)
            bucket["breaths"] += breaths
            bucket["seconds"] += durations[row_index] or 0.0

        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), OUT / path.name)
        if position % 25 == 0:
            print(f"    {position}/{len(shards)} IV-R shards", flush=True)

    rows = totals["rows"]
    join_rate = totals["joined"] / max(1, rows)
    event_rate = totals["with_events"] / max(1, rows)
    marked_hours = totals["seconds"] / 3600.0

    lines = [
        "# ETL step 3b: IV-R event marks recovered from IndicVoices",
        "",
        f"- IV-R training rows (leak-filtered, step 2): **{rows:,}**",
        f"- joined to an IndicVoices twin: **{totals['joined']:,} ({join_rate:.1%})**",
        f"- twin carries at least one voiceable event: **{totals['with_events']:,} ({event_rate:.1%})**",
        f"- breath marks recovered: **{totals['breaths']:,}**; all voiceable events: {totals['events']:,}",
        f"- hours represented by these rows: **{marked_hours:.1f} h**",
        f"- ambiguous join keys in the IndicVoices index: {index.ambiguous:,}",
        "",
        "Only *voiceable* events are counted: fillers and environment noise map to no",
        "model token and are excluded (see `geetha.text.contract.map_corpus_nonverbal`).",
        "",
        "## Per task_name",
        "",
        "| task_name | rows | joined | with events | breaths | hours |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, bucket in sorted(per_task.items(), key=lambda item: -item[1]["rows"]):
        lines.append(
            f"| {task} | {bucket['rows']:,} | {bucket['joined'] / max(1, bucket['rows']):.1%} "
            f"| {bucket['with_events'] / max(1, bucket['rows']):.1%} | {bucket['breaths']:,} "
            f"| {bucket['seconds'] / 3600.0:.1f} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"rows {rows:,} | joined {join_rate:.1%} | with events {event_rate:.1%} "
          f"| breaths {totals['breaths']:,} | {marked_hours:.1f} h")
    print(f"wrote {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
