r"""Walk parquet shards and measure every clip -- the driver for ETL step 1.

Three properties this has to have, each learned from something that already went wrong
on this project:

**Resumable.** E: is an exFAT external drive that has disconnected mid-session twice
(28 Oct, 3 Nov). A scan of 137 shards that has to start over because it died on shard
120 is a scan that never finishes. Output is one parquet per input shard, and a shard
whose output exists is skipped, so the work survives the drive.

**Survives bad data.** A corrupt audio blob is recorded as a row with an ``error`` and
null measurements, not raised: one unreadable clip in 39,292 must not cost the other
39,291. An unreadable *shard* is skipped with a warning and leaves no output file, so a
later run retries it rather than inheriting a half-written answer.

**Keeps both durations.** The stored ``duration`` column is carried through beside the
duration decoded from the samples. Trusting the metadata is how you discover at
training time that it was wrong; carrying both makes the disagreement a column you can
sort by.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from geetha.audio.qc import clip_qc

__all__ = ["ShardSpec", "qc_shard", "scan_shards"]

# Columns every row carries, whether the clip decoded or not.
_MEASURED = (
    "duration_s", "peak", "rms_dbfs", "clipped_samples", "clipped_fraction",
    "leading_silence_s", "trailing_silence_s", "is_silent", "sample_rate", "channels",
)


@dataclass(frozen=True)
class ShardSpec:
    """How to read one corpus: which column holds the audio, what to carry through."""

    source: str
    audio_column: str
    carry: tuple[str, ...] = ()
    duration_column: str = "duration"


def _schema(spec: ShardSpec) -> pa.Schema:
    fields = [
        pa.field("source", pa.string()),
        pa.field("shard", pa.string()),
        pa.field("row_index", pa.int32()),
        pa.field("clip_path", pa.string()),
        pa.field("duration_meta_s", pa.float64()),
        pa.field("duration_s", pa.float64()),
        pa.field("peak", pa.float64()),
        pa.field("rms_dbfs", pa.float64()),
        pa.field("clipped_samples", pa.int64()),
        pa.field("clipped_fraction", pa.float64()),
        pa.field("leading_silence_s", pa.float64()),
        pa.field("trailing_silence_s", pa.float64()),
        pa.field("is_silent", pa.bool_()),
        pa.field("sample_rate", pa.int32()),
        pa.field("channels", pa.int32()),
        pa.field("error", pa.string()),
    ]
    fields += [pa.field(name, pa.string()) for name in spec.carry]
    return pa.schema(fields)


def _as_float(value: object) -> float | None:
    """Rasa stores duration as a string; IV-R as a double."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def qc_shard(path: Path | str, spec: ShardSpec) -> pa.Table:
    """Measure every clip in one shard. Never raises for a single bad clip."""
    path = Path(path)
    available = set(pq.ParquetFile(path).schema_arrow.names)

    # Check names against the schema before reading: pyarrow's own error for an absent
    # column dumps the whole struct schema and buries which name was wrong.
    missing = [name for name in (spec.audio_column, *spec.carry) if name not in available]
    if missing:
        raise KeyError(f"{path.name} has no column(s): {', '.join(missing)}")

    wanted = [spec.audio_column, *spec.carry]
    if spec.duration_column in available:
        wanted.append(spec.duration_column)
    table = pq.read_table(path, columns=wanted)

    audio = table[spec.audio_column].to_pylist()
    stored = (table[spec.duration_column].to_pylist()
              if spec.duration_column in table.column_names else [None] * len(audio))
    carried = {name: table[name].to_pylist() for name in spec.carry}

    rows: list[dict[str, object]] = []
    for index, blob in enumerate(audio):
        row: dict[str, object] = {
            "source": spec.source,
            "shard": path.name,
            "row_index": index,
            "clip_path": (blob or {}).get("path") or "",
            "duration_meta_s": _as_float(stored[index]),
            "error": "",
        }
        row.update({name: None for name in _MEASURED})
        try:
            wave, sample_rate = sf.read(io.BytesIO((blob or {})["bytes"]), dtype="float32", always_2d=True)
            measured = clip_qc(wave, sample_rate)
            row.update(measured.as_row())
            row["sample_rate"] = sample_rate
            row["channels"] = wave.shape[1]
        except Exception as problem:  # one bad clip must not cost the shard
            row["error"] = f"{type(problem).__name__}: {problem}"[:300]
        for name, values in carried.items():
            value = values[index]
            row[name] = None if value is None else str(value)
        rows.append(row)

    return pa.Table.from_pylist(rows, schema=_schema(spec))


def scan_shards(
    shards: Sequence[Path | str],
    spec: ShardSpec,
    out_dir: Path | str,
    *,
    force: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Measure each shard into ``out_dir``, skipping those already done.

    Returns the outputs written by *this* call, so an empty list means "everything
    was already done" rather than "nothing happened".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for position, shard in enumerate(shards, start=1):
        shard = Path(shard)
        target = out_dir / shard.name
        if target.exists() and not force:
            continue
        try:
            table = qc_shard(shard, spec)
        except Exception as problem:
            # Leave no output, so the next run retries this shard.
            print(f"  !! {shard.name}: {type(problem).__name__}: {problem}", file=sys.stderr)
            continue
        # Write beside the target and move, so a kill mid-write cannot leave a
        # truncated file that a later run would treat as done.
        staging = target.with_suffix(".parquet.part")
        pq.write_table(table, staging)
        staging.replace(target)
        written.append(target)
        if progress:
            print(f"  [{position}/{len(shards)}] {shard.name}: {table.num_rows} clips", flush=True)

    return written
