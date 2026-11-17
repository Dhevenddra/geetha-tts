r"""B5 part 2 — freeze the authored prompt sets (runbook section 5, R13).

    py -3.11 scripts\freeze_eval_vault_part2.py

Part 1 froze the public benchmark corpora and refs_v1 on 27 Oct and said, in its own
closing line, that part 2 is "a separate run on its own date — do not backdate it".
This is that run, on 5 Nov 2026, and the date in the CSV is today's for that reason.

What part 2 covers: the prompt sets we authored or selected ourselves. These are not
downloadable, so nothing but this hash distinguishes the set a number was measured on
from an edited one.

  PROMPTS  a prompt set a model renders and is scored on
  AUDIO    real recordings that are half of a listening pair

Why realness is only being frozen now
-------------------------------------
``realness_ta_v1.jsonl`` did not exist until today. ``make_naturalness_ta.py`` builds
the CAPABILITY set and says so in its own docstring; the 40 real-vs-rendered Rasa pairs
-- the project's HEADLINE metric -- had no prompt set at all. It was built by
``make_naturalness_realness.py`` and frozen here in the same session, before any number
has been measured on it, which is the only ordering that makes the freeze mean anything.

The capability set has existed since 27 Oct but was never frozen either: part 2 was
deferred and then not run. Both go in together.

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

PART = 2
FROZEN_ON = date.today().isoformat()

VAULT = "eval_vault/naturalness_ta_v1"

# (asset, kind, [globs relative to ROOT])
ASSETS: list[tuple[str, str, list[str]]] = [
    # --- Naturalness-Ta REALNESS: the headline metric (runbook section 6)
    # The jsonl and the audio are one asset in two kinds: a pair is only reproducible
    # if both the sentence list AND the exact real recording are pinned. Swap the wav
    # and the same jsonl measures something else.
    ("realness_ta_v1", "PROMPTS", [f"{VAULT}/realness_ta_v1.jsonl"]),
    ("realness_ta_v1_audio", "AUDIO", [f"{VAULT}/realness_audio/*.wav"]),
    # --- Naturalness-Ta CAPABILITY, authored 27 Oct, frozen now
    ("capability_ta_v1", "PROMPTS", [f"{VAULT}/capability_ta_v1.jsonl"]),
    # --- Directability-Ta (B4), the direction-tag prompt set
    ("directability_ta_v1", "PROMPTS", ["eval_vault/directability_ta_v1.jsonl"]),
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
        megabytes = sum(row.bytes for row in asset_rows) / 1e6
        print(f"  {asset:28s} {len(asset_rows):3d} files  {megabytes:8.2f} MB", flush=True)
        rows.extend(asset_rows)

    merged = merge_rows(existing, rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({field: getattr(row, field) for field in FIELDS})

    total = sum(row.bytes for row in rows) / 1e6
    print(f"\npart {PART} frozen {FROZEN_ON}: {len(rows)} files, {total:.1f} MB -> {OUT}")
    print("verify with: py -3.11 scripts\\verify_eval_freeze.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
