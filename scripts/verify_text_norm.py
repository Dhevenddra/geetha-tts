r"""Check ETL step 2's output: the leak table must be zero.

    py -3.11 scripts\verify_text_norm.py

Runbook section 5 step 2 is "done when tests pass; leak table = 0". The tests cover the
decision logic; this covers the **artefacts on disk**. It re-reads every held-out set,
re-reads what step 2 actually wrote, and asserts that no row marked ``keep`` carries a
normalised text belonging to a held-out split.

That is deliberately a different question from "did the code work". Step 2 could be
correct and the files still wrong -- a half-written shard from a dropped drive, a stale
output from before a prompt set changed, a shard the glob missed. This reads the files.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.data.holdout import HoldoutIndex  # noqa: E402
from run_text_norm import OUT, TRAINING, holdout_entries, speaker_holdout  # noqa: E402


def main() -> int:
    index = HoldoutIndex(holdout_entries())
    held_out, too_small = speaker_holdout()
    print(f"holdout index: {len(index):,} sentences; {len(held_out)} held-out speakers")

    failures: list[str] = []
    grand_kept = 0

    for source, directory, pattern, _text, _speaker in TRAINING:
        expected = {path.name for path in directory.glob(pattern)}
        written = {path.name for path in (OUT / source).glob("*.parquet")}
        if missing := sorted(expected - written):
            failures.append(f"{source}: {len(missing)} input shards have no output, "
                            f"first is {missing[0]}")
        if extra := sorted(written - expected):
            failures.append(f"{source}: {len(extra)} outputs have no input shard, "
                            f"first is {extra[0]}")

        kept = 0
        leaked: Counter[str] = Counter()
        bad_speakers: Counter[str] = Counter()
        for path in sorted((OUT / source).glob("*.parquet")):
            table = pq.read_table(path, columns=["speaker_id", "norm_text", "keep", "reason"])
            for row in table.to_pylist():
                if not row["keep"]:
                    continue
                kept += 1
                if row["reason"]:
                    failures.append(f"{source}/{path.name}: a kept row carries reason "
                                    f"{row['reason']!r}")
                # norm_text is already normalised, so look it up as-is.
                source_set = index.source_of(row["norm_text"])
                if source_set is not None:
                    leaked[source_set] += 1
                if row["speaker_id"] in held_out:
                    bad_speakers["held_out"] += 1
                elif row["speaker_id"] in too_small:
                    bad_speakers["too_small"] += 1

        grand_kept += kept
        verdict = "ok" if not leaked and not bad_speakers else "FAIL"
        print(f"  {verdict:4s} {source}: {kept:,} kept rows, "
              f"{sum(leaked.values())} leaked, {sum(bad_speakers.values())} from held-out speakers")
        for name, count in leaked.most_common():
            failures.append(f"{source}: {count} kept rows collide with {name}")
        for name, count in bad_speakers.most_common():
            failures.append(f"{source}: {count} kept rows belong to {name} speakers")

    print()
    if failures:
        print("LEAK TABLE IS NOT ZERO:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"leak table = 0. {grand_kept:,} trainable rows across "
          f"{len(TRAINING)} sources, no held-out text and no held-out speaker among them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
