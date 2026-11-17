"""Reading and appending manifests/burn_tracker.csv.

The tracker is the record of every rupee: a `planned` row before a launch, the
actual cost after. ``read_burn_tracker`` is strict on purpose -- a tracker that
cannot be parsed has to block a launch, because "no rows found" and "I could not
read the file" must never look the same to the guards.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from ops.guards import BurnRow

__all__ = [
    "BURN_COLUMNS",
    "BURN_STATUSES",
    "TERMINAL_BURN_STATUSES",
    "BurnTrackerError",
    "read_burn_tracker",
    "set_status",
]

BURN_COLUMNS = (
    "date",
    "gate",
    "run_id",
    "provider",
    "gpu",
    "planned_hours",
    "actual_hours",
    "rate_usd_per_h",
    "inr_per_usd",
    "max_cost_inr",
    "cost_inr",
    "cumulative_inr",
    "status",
    "purpose",
    "result_path",
)

BURN_STATUSES = ("planned", "running", "done", "aborted")

# A row moves forward only. `planned` authorises exactly one launch, which consumes it
# by moving it to `running`; `done` and `aborted` are final. Without this, destroying a
# run left its row on `planned` and one gate approval permitted unlimited relaunches
# (found by security review, 27 Oct).
ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "planned": ("running", "aborted"),
    "running": ("done", "aborted"),
    "done": (),
    "aborted": (),
}

# Derived, never written out by hand: a status is terminal exactly when nothing can
# follow it. Listing them separately would let the two drift, and a status wrongly
# believed non-terminal is one a caller will try to overwrite.
TERMINAL_BURN_STATUSES: frozenset[str] = frozenset(
    status for status, nexts in ALLOWED_TRANSITIONS.items() if not nexts
)


class BurnTrackerError(ValueError):
    """A burn-tracker write that must not happen."""


def _number(value: str, default: float = 0.0) -> float:
    value = (value or "").strip()
    return float(value) if value else default


def read_burn_tracker(path: str | Path) -> list[BurnRow]:
    """Parse the burn tracker. Raises on anything it does not understand."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"burn tracker not found at {path}: a launch cannot be authorised without it"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != BURN_COLUMNS:
            raise ValueError(
                f"burn tracker column mismatch.\n  expected: {', '.join(BURN_COLUMNS)}\n"
                f"  found:    {', '.join(header)}"
            )
        rows: list[BurnRow] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            run_id = (raw["run_id"] or "").strip()
            if not run_id:
                continue
            status = (raw["status"] or "").strip()
            if status not in BURN_STATUSES:
                raise ValueError(
                    f"burn tracker line {line_number}: status {status!r} is not one of "
                    f"{', '.join(BURN_STATUSES)}"
                )
            if run_id in seen:
                raise ValueError(f"burn tracker line {line_number}: duplicate run_id {run_id!r}")
            seen.add(run_id)
            rows.append(
                BurnRow(
                    run_id=run_id,
                    gate=(raw["gate"] or "").strip(),
                    status=status,
                    planned_hours=_number(raw["planned_hours"]),
                    rate_usd_per_h=_number(raw["rate_usd_per_h"]),
                    cost_inr=_number(raw["cost_inr"]),
                    max_cost_inr=_number(raw["max_cost_inr"]),
                )
            )
    return rows


def set_status(
    path: str | Path,
    run_id: str,
    status: str,
    *,
    actual_hours: float | None = None,
    cost_inr: float | None = None,
) -> None:
    """Move one run's row to ``status``, optionally recording what it actually cost.

    Forward-only (see ``ALLOWED_TRANSITIONS``). An unknown run_id raises rather than
    quietly doing nothing: a silent no-op would let a launch believe its authorisation
    had been consumed when it had not.

    The cumulative column is recomputed for every row on each write, so it cannot drift
    out of step with the costs above it.
    """
    path = Path(path)
    if status not in BURN_STATUSES:
        raise BurnTrackerError(
            f"status {status!r} is not one of {', '.join(BURN_STATUSES)}"
        )

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != BURN_COLUMNS:
            raise BurnTrackerError(
                f"burn tracker column mismatch.\n"
                f"  expected: {', '.join(BURN_COLUMNS)}\n"
                f"  found:    {', '.join(header)}"
            )
        rows = list(reader)

    matches = [row for row in rows if (row["run_id"] or "").strip() == run_id]
    if not matches:
        raise BurnTrackerError(
            f"{run_id!r} is not in the burn tracker, so there is nothing to mark "
            f"{status!r}. Write the `planned` row first."
        )
    row = matches[0]
    current = (row["status"] or "").strip()
    if status not in ALLOWED_TRANSITIONS.get(current, ()):
        raise BurnTrackerError(
            f"{run_id!r} is {current!r} and cannot become {status!r} "
            f"(allowed from {current!r}: {ALLOWED_TRANSITIONS.get(current, ()) or 'nothing'}). "
            f"A row moves forward only, so one approval authorises one launch."
        )

    row["status"] = status
    if actual_hours is not None:
        row["actual_hours"] = f"{actual_hours:.2f}"
    if cost_inr is not None:
        row["cost_inr"] = f"{cost_inr:.2f}"

    running_total = 0.0
    for each in rows:
        running_total += _number(each.get("cost_inr", ""))
        each["cumulative_inr"] = f"{running_total:.2f}"

    # Written to a temporary file in the same directory and moved into place, so a
    # failure part-way through leaves the old tracker intact. Rewriting in place would
    # truncate the record of every rupee already spent; later launches would then
    # refuse (fail-closed, which is right) but the history would be gone.
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False,
        dir=str(path.parent), prefix=path.name, suffix=".tmp",
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(BURN_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
