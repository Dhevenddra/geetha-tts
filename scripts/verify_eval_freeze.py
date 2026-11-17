r"""Re-hash everything the eval freeze recorded, and say what moved.

    py -3.11 scripts\verify_eval_freeze.py [--quick]

``--quick`` checks only files under 100 MB -- the derived references and prompt sets --
which takes seconds. Without it, all 37.7 GB is re-hashed.

Why this exists beyond the freeze itself
----------------------------------------
B5 froze 25 files by SHA-256 so that a published number can be tied to the exact bytes
it was measured on. That same list is a **filesystem integrity check**, and on 27 Oct
2026 it earned its keep: E: is an exFAT external drive that disconnected and reconnected
mid-session, and Windows reported ``Full Repair Needed``.

A dirty exFAT volume does not announce which files it damaged. Silent corruption in a
test set is the worst possible failure for this project, because it does not crash
anything -- it quietly changes a benchmark, and every number measured afterwards is
wrong in a way no reviewer could detect. The freeze turns that from an unknowable risk
into a list.

A mismatch here means: do not train, do not publish, restore the file and re-freeze it
under a new asset name.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.eval.freeze import hash_file  # noqa: E402

ROOT = Path(r"E:\Geetha")
FREEZE = ROOT / "manifests" / "eval_freeze.csv"
QUICK_LIMIT = 100_000_000


def main() -> int:
    quick = "--quick" in sys.argv
    if not FREEZE.exists():
        print(f"no freeze manifest at {FREEZE}")
        return 1

    with FREEZE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    checked = skipped = 0
    missing: list[str] = []
    changed: list[str] = []
    resized: list[str] = []

    for row in rows:
        path = ROOT / row["path"]
        expected_bytes = int(row["bytes"])
        if not path.exists():
            missing.append(row["path"])
            continue
        if quick and expected_bytes > QUICK_LIMIT:
            skipped += 1
            continue

        actual_size = path.stat().st_size
        if actual_size != expected_bytes:
            resized.append(f"{row['path']}  {expected_bytes} -> {actual_size}")
            continue

        digest, _ = hash_file(path)
        checked += 1
        if digest != row["sha256"]:
            changed.append(row["path"])
        print(f"  {'ok ' if digest == row['sha256'] else 'BAD'}  {row['asset']:28s} {row['path']}",
              flush=True)

    print(f"\n{checked} verified, {skipped} skipped" + (" (--quick)" if quick else ""))
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
    if resized:
        print(f"\nWRONG SIZE ({len(resized)}):")
        for name in resized:
            print(f"  {name}")
    if changed:
        print(f"\nCONTENT CHANGED ({len(changed)}):")
        for name in changed:
            print(f"  {name}")

    if missing or resized or changed:
        print("\nDo not train or publish on these. Restore the file, then re-freeze it")
        print("under a NEW asset name so the manifest shows the substitution.")
        return 2

    # The date was hardcoded when 27 Oct was the only freeze date. Part 2 landed on
    # 5 Nov, so a fixed string here would report a date that is simply false -- and a
    # verifier that misstates what it verified is worse than no verifier.
    dates = sorted({row["frozen_on"] for row in rows})
    when = dates[0] if len(dates) == 1 else f"{dates[0]}..{dates[-1]}"
    print(f"\nEvery frozen file matches the hash recorded on {when}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
