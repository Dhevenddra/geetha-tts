r"""Build Stage B's codes from Stage A's, rather than encoding the audio twice.

    py -3.11 scripts\derive_stageb_codes.py

Every Stage B row *is* a Stage A row: step 5 selects Rasa's CONV and emotion clips out
of the same set, changing only the ``repeat`` count. Same clip, same loudness gain, same
preparation -- so the codes are the same codes, and encoding them again would be 20.3 h
of audio through the GPU for an identical answer.

Measured first rather than assumed: SNAC's **encode** is deterministic (three runs on
the same clip gave byte-identical codes -- quantisation to discrete codes is stable
under small float differences), even though its **decode** is not reproducible on CUDA.
So deriving and re-encoding would agree; deriving is simply free, and it guarantees the
two stages cannot drift apart.

Output: ``processed\tts_codes\stageB\*.parquet`` plus ``checksums.csv``, in the same
schema the encoder writes, with ``repeat`` taken from the Stage B manifest.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(r"E:\Geetha")
CODES = ROOT / "processed" / "tts_codes"
STAGE_B_MANIFEST = ROOT / "processed" / "tts_stageB" / "manifest.parquet"


def main() -> int:
    source_dir = CODES / "stageA"
    out_dir = CODES / "stageB"
    shards = sorted(source_dir.glob("*.parquet"))
    if not shards:
        print(f"!! nothing encoded in {source_dir}; run run_snac_encode.py --stage A first",
              file=sys.stderr)
        return 1

    wanted: dict[tuple[str, str, int], int] = {}
    for row in pq.read_table(STAGE_B_MANIFEST).to_pylist():
        wanted[(row["source"], row["shard"], row["row_index"])] = row["repeat"]
    print(f"Stage B manifest: {len(wanted):,} clips")

    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    schema = pq.read_schema(shards[0])
    found = 0

    for path in shards:
        table = pq.read_table(path)
        for row in table.to_pylist():
            key = (row["source"], row["shard"], row["row_index"])
            repeat = wanted.get(key)
            if repeat is None:
                continue
            grouped[path.name].append({**row, "repeat": repeat})
            found += 1

    if found != len(wanted):
        print(f"!! Stage B wants {len(wanted):,} clips but only {found:,} are present in "
              f"Stage A's codes. Finish the Stage A encode before deriving.", file=sys.stderr)
        return 1

    for name, rows in sorted(grouped.items()):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_dir / name)

    lines = ["shard,sha256,bytes"]
    total_tokens = weighted = 0
    for path in sorted(out_dir.glob("*.parquet")):
        lines.append(f"{path.name},{hashlib.sha256(path.read_bytes()).hexdigest()},"
                     f"{path.stat().st_size}")
    for rows in grouped.values():
        for row in rows:
            total_tokens += row["n_tokens"]
            weighted += row["n_tokens"] * row["repeat"]
    (out_dir / "checksums.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Stage B: {found:,} clips over {len(grouped)} shards, "
          f"{total_tokens:,} tokens ({weighted:,} with repeats)")
    print(f"checksums: {len(lines) - 1} shards -> {out_dir / 'checksums.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
