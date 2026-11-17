r"""B5 part 1 — freeze the benchmark corpora and refs_v1 (runbook section 5, R13).

    py -3.11 scripts\freeze_eval_vault.py

Writes ``E:\Geetha\manifests\eval_freeze.csv``. Part 2 — the authored prompt sets — is a
separate run on its own honest date, once the operator has approved the Tamil and the
contamination check has settled where Naturalness-Ta comes from.

The asset list below is runbook section 6's evaluation suite, resolved to the files that
actually exist on E:. Where section 6 names a *sample* of a corpus rather than the whole
thing -- the 2,000-clip Common Voice test, seed 1234 -- the source archive is what gets
hashed here. The sample is derived from it deterministically, so hashing the source
pins the sample too, and the sampling code is tagged ``eval-v1`` alongside.

Nothing in ``raw\`` is modified. This reads and hashes only.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.eval.freeze import FIELDS, FrozenFile, freeze_rows, merge_rows  # noqa: E402

ROOT = Path(r"E:\Geetha")
OUT = ROOT / "manifests" / "eval_freeze.csv"

PART = 1
FROZEN_ON = date.today().isoformat()

# (asset, kind, [globs relative to ROOT])
#
# BENCH   a public test set we re-run and publish numbers on
# REFS    our own derived references and lexicon — scorable, and the thing a reader
#         needs in order to reproduce a CER number exactly
ASSETS: list[tuple[str, str, list[str]]] = [
    # --- ASR benchmarks (runbook section 6, "ASR")
    ("fleurs_ta_test", "BENCH", ["raw/hf/fleurs/parquet-data/ta_in/test-*.parquet"]),
    ("spring_inx_r1_test", "BENCH", ["raw/hf/spring_inx_tamil_r1/data/test-*.parquet"]),
    ("spring_inx_r2_test", "BENCH", ["raw/hf/spring_inx_tamil_r2/data/test-*.parquet"]),
    ("indicvoices_ta_valid", "BENCH", ["raw/hf/indicvoices/tamil/valid-*.parquet"]),
    ("kathbath_ta_valid", "BENCH", ["raw/hf/kathbath/tamil/valid-*.parquet"]),
    ("kathbath_tests", "BENCH", ["raw/http/kathbath_tests/*.tar"]),
    ("openslr127_test", "BENCH", ["raw/http/openslr127_iisc_mile_tamil/*.tar.gz"]),
    ("common_voice_27_ta", "BENCH", ["raw/mdc/common_voice_27_ta/*.tar.gz"]),
    ("indic_diarbench_ta", "BENCH", ["eval_vault/indic_diarbench/Tamil/test-*.parquet"]),
    ("sarvam_contextual_asr_ta", "BENCH", ["eval_vault/sarvam_contextual_asr/ta-IN/test-*.parquet"]),
    # --- TTS prompt benchmarks (runbook section 6, "Robustness")
    ("sarvam_tts_general", "BENCH", ["eval_vault/sarvam_tts_general_benchmark/data/*.parquet"]),
    ("sarvam_tts_robustness", "BENCH", ["eval_vault/sarvam_tts_robustness_benchmark/data/*.parquet"]),
    # --- our derived references (B3)
    ("refs_v1", "REFS", ["eval_vault/refs_v1/*.jsonl", "eval_vault/refs_v1/lexicon_ta_en.json"]),
]


def read_existing() -> list[FrozenFile]:
    if not OUT.exists():
        return []
    with OUT.open(encoding="utf-8", newline="") as handle:
        return [
            FrozenFile(
                frozen_on=row["frozen_on"],
                part=int(row["part"]),
                asset=row["asset"],
                kind=row["kind"],
                path=row["path"],
                sha256=row["sha256"],
                bytes=int(row["bytes"]),
            )
            for row in csv.DictReader(handle)
        ]


def main() -> int:
    existing = read_existing()
    if existing:
        print(f"{OUT.name} already holds {len(existing)} rows; adding to it")

    rows: list[FrozenFile] = []
    for asset, kind, globs in ASSETS:
        paths: list[Path] = []
        for pattern in globs:
            paths.extend(ROOT.glob(pattern))
        if not paths:
            print(f"  !! {asset}: no files matched {globs} — SKIPPED, investigate")
            continue
        asset_rows = freeze_rows(
            asset, kind, paths, part=PART, frozen_on=FROZEN_ON, root=ROOT,
            existing=[*existing, *rows],
        )
        gigabytes = sum(row.bytes for row in asset_rows) / 1e9
        print(f"  {asset:28s} {len(asset_rows):3d} files  {gigabytes:7.2f} GB", flush=True)
        rows.extend(asset_rows)

    merged = merge_rows(existing, rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({field: getattr(row, field) for field in FIELDS})

    total = sum(row.bytes for row in rows) / 1e9
    print(f"\npart {PART} frozen {FROZEN_ON}: {len(rows)} files, {total:.1f} GB -> {OUT}")
    print("part 2 (authored prompt sets) is a separate run on its own date — do not backdate it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
