r"""The machine-readable record of what the operator actually approved.

    E:\Geetha\manifests\gate_approvals.csv

Why this file exists at all
---------------------------
``check_launch`` has always required a ``GateApproval`` and nothing could produce one,
so ``gpu.py launch`` refused unconditionally. That was the right default while the
safety machinery was unfinished, but it is not a safety property -- it is an unfinished
feature, and an unfinished feature is the sort of thing that gets worked around at
11pm before a gate.

The runbook says the approval lives in ``DECISIONS.md``, which means parsing prose to
decide whether money may be spent. Proposal R11 argued for a machine-readable file
instead, and D-014 accepted it in principle. So: ``DECISIONS.md`` stays the human
record of *why*, and this CSV is the machine record of *how much*. The ``/gate`` flow
writes a row here only after the operator's literal "approved".

The part that makes it stronger than prose
------------------------------------------
Every ceiling is cross-checked against the gate budget the plan authorised (D-013). If
a row claims more than its gate was ever allocated, the loader refuses the whole file
rather than honouring the larger number. So even a row written in error -- by me,
mis-transcribing a conversation -- cannot authorise more than the plan did. A
mis-recorded approval is then a *refused launch*, which is the direction an error in
this file has to fail.

For the same reason: two rows for one gate are refused outright rather than resolved
"last wins". Two ceilings in force means nobody knows what the limit is.
"""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

from ops.guards import GateApproval

__all__ = [
    "APPROVAL_COLUMNS",
    "APPROVALS_PATH",
    "GATE_BUDGETS",
    "ApprovalError",
    "append_approval",
    "read_approvals",
    "total_approved_inr",
]

APPROVALS_PATH = Path(r"E:\Geetha\manifests\gate_approvals.csv")

APPROVAL_COLUMNS = ("date", "gate", "max_inr", "approved_by", "note")

# D-013. A row may approve less than its gate's budget; it may never approve more.
GATE_BUDGETS: dict[str, float] = {
    "M1": 300.0,
    "M2": 1000.0,
    "M3": 500.0,
    "M4": 1300.0,
    "M5": 600.0,
    "M6": 600.0,
}

# One person on this project approves money (the project rules). The column is not decoration.
APPROVER = "operator"

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ApprovalError(ValueError):
    """An approvals file that must not be used to authorise a launch."""


def read_approvals(path: str | Path = APPROVALS_PATH) -> dict[str, GateApproval]:
    """Load the approvals file, keyed by gate.

    A missing file is not an error: nothing approved yet is this project's normal
    state, and it must read as "refuse every launch" rather than as a crash someone is
    tempted to route around.
    """
    path = Path(path)
    if not path.exists():
        return {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != APPROVAL_COLUMNS:
            raise ApprovalError(
                f"gate approvals column mismatch.\n"
                f"  expected: {', '.join(APPROVAL_COLUMNS)}\n"
                f"  found:    {', '.join(header)}"
            )

        approvals: dict[str, GateApproval] = {}
        for line_number, raw in enumerate(reader, start=2):
            gate = (raw["gate"] or "").strip()
            if not gate:
                continue
            where = f"gate approvals line {line_number}"

            if gate not in GATE_BUDGETS:
                raise ApprovalError(
                    f"{where}: {gate!r} is not a gate ({', '.join(GATE_BUDGETS)})"
                )
            if gate in approvals:
                raise ApprovalError(
                    f"{where}: duplicate row for {gate}. Two ceilings in force means "
                    f"nobody knows the limit — delete one."
                )

            when = (raw["date"] or "").strip()
            if not _ISO_DATE.match(when):
                raise ApprovalError(
                    f"{where}: date {when!r} must be YYYY-MM-DD. An undated "
                    f"authorisation is the kind that stays valid forever."
                )

            approved_by = (raw["approved_by"] or "").strip().lower()
            if approved_by != APPROVER:
                raise ApprovalError(
                    f"{where}: approved_by is {approved_by!r}, not {APPROVER!r}. "
                    f"Only the operator approves money."
                )

            try:
                max_inr = float((raw["max_inr"] or "").strip())
            except ValueError:
                raise ApprovalError(
                    f"{where}: max_inr {raw['max_inr']!r} is not a number"
                ) from None
            if max_inr <= 0:
                raise ApprovalError(f"{where}: max_inr must be positive, got {max_inr}")
            budget = GATE_BUDGETS[gate]
            if max_inr > budget:
                raise ApprovalError(
                    f"{where}: INR {max_inr:.0f} is above the {gate} budget of "
                    f"INR {budget:.0f} (D-013). Refusing the whole file rather than "
                    f"honouring the larger number."
                )

            approvals[gate] = GateApproval(gate=gate, approved=True, max_inr=max_inr)
    return approvals


def total_approved_inr(approvals: dict[str, GateApproval]) -> float:
    return sum(a.max_inr for a in approvals.values() if a.approved)


def append_approval(
    gate: str,
    max_inr: float,
    *,
    note: str = "",
    path: str | Path = APPROVALS_PATH,
    when: date | None = None,
) -> None:
    """Record one approval. Called by the ``/gate`` flow after the literal "approved".

    Re-reads the file afterwards, so a row that would make the file unloadable fails
    here rather than at the next launch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(APPROVAL_COLUMNS)
        writer.writerow([
            (when or date.today()).isoformat(),
            gate,
            f"{max_inr:.0f}",
            APPROVER,
            note,
        ])
    read_approvals(path)
