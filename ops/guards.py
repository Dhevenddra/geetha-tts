"""The rules that stand between Geetha and a surprise bill.

Everything here is a pure function over plain data, so every refusal can be tested
without a provider account and without spending anything. The network layer calls
these; it never re-implements them.

Launch is refused unless all of these hold (runbook section 9):

1. the gate named by the spec has an approval recorded;
2. the spec's rupee ceiling is within what that gate authorised;
2b. the spec's ceiling *plus every other run already committed under that gate* is
    still within what the gate authorised -- a gate carries several run_ids, so
    checking one spec at a time would let three INR 300 runs through a INR 300 gate;
3. a `planned` row for this run_id exists in the burn tracker, under the same gate,
   carrying a ceiling at least as large as the spec's -- otherwise the exposure sum in
   2b is computed from a number the spec is free to exceed;
4. the prepaid balance covers the ceiling -- the balance is the hard ceiling;
5. the quoted hourly price is at most 20% above the planned rate;
6. the full run at the quoted price still fits under the spec's rupee ceiling;
7. no pod for this run_id is already alive.

``pods_to_destroy`` is what the watchdog uses. It destroys on cost as well as on
time, because the deadline alone does not bound the bill if the hourly price moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

__all__ = [
    "BurnRow",
    "GateApproval",
    "LaunchDecision",
    "LaunchRefused",
    "PodState",
    "PRICE_TOLERANCE",
    "check_launch",
    "gate_exposure_inr",
    "pods_to_destroy",
]

# "price <= plan +20%" (runbook section 9). Inclusive: exactly +20% is allowed.
PRICE_TOLERANCE = 1.20


class LaunchRefused(Exception):
    """A launch that must not happen, with the reason in the message."""


@dataclass(frozen=True)
class GateApproval:
    """One money gate's state.

    Loaded from ``manifests/gate_approvals.csv`` by ``ops.approvals`` (proposal R11,
    accepted in principle by D-014). DECISIONS.md remains the human record of *why* a
    gate was approved; that file is the machine record of *how much*. These rules work
    with either source, and never with a ceiling the loader has not cross-checked
    against the gate budget D-013 authorised.
    """

    gate: str
    approved: bool
    max_inr: float


@dataclass(frozen=True)
class BurnRow:
    """One row of manifests/burn_tracker.csv."""

    run_id: str
    gate: str
    status: str  # planned | running | done | aborted
    planned_hours: float
    rate_usd_per_h: float
    cost_inr: float
    # The ceiling this run is authorised to reach, mirroring its job spec. Exposure is
    # computed on ceilings, not on expected cost: five runs each *planning* INR 10 but
    # *authorised* to INR 300 commit INR 1,500 of a gate, not INR 50.
    max_cost_inr: float = 0.0


@dataclass(frozen=True)
class PodState:
    """A live pod as the provider reports it, plus the ceilings set at launch."""

    pod_id: str
    run_id: str
    hours_used: float
    cost_inr: float
    max_cost_inr: float
    deadline: datetime


@dataclass(frozen=True)
class LaunchDecision:
    """An allowed launch and what it is expected to cost."""

    spec_run_id: str
    estimated_cost_inr: float
    quoted_rate_usd_per_h: float
    inr_per_usd: float


def row_exposure_inr(row: "BurnRow", inr_per_usd: float) -> float:
    """What one burn-tracker row has committed of its gate's authorisation.

    A finished run has consumed exactly what it cost. A planned or running one has
    committed whichever is largest of: the ceiling it is authorised to reach, what it
    was expected to cost, and what it has already spent.

    The ceiling is the honest figure, because that is what a launch is allowed to run
    up to. Using expected cost alone reopened the hole this function exists to close:
    several runs each planning a little but authorised to a lot would pass one at a
    time. ``max_cost_inr`` defaults to 0 for a row written before the column existed,
    and the planned estimate carries those.
    """
    planned = row.planned_hours * row.rate_usd_per_h * inr_per_usd
    if row.status in ("done", "aborted"):
        return row.cost_inr
    return max(row.cost_inr, planned, row.max_cost_inr)


def gate_exposure_inr(
    burn_rows: Sequence["BurnRow"],
    *,
    gate: str,
    inr_per_usd: float,
    excluding: str = "",
) -> float:
    """Total committed against one gate, optionally ignoring one run_id.

    ``excluding`` is the run being launched: its own planned row is the authorisation
    being used, not prior exposure, so counting it would refuse every first launch.
    """
    return sum(
        row_exposure_inr(row, inr_per_usd)
        for row in burn_rows
        if row.gate == gate and row.run_id != excluding
    )


def check_launch(
    *,
    spec,
    approval: GateApproval | None,
    burn_rows: Sequence[BurnRow],
    balance_inr: float,
    quoted_rate_usd_per_h: float,
    inr_per_usd: float,
    running_pods: Sequence[PodState],
) -> LaunchDecision:
    """Return a ``LaunchDecision``, or raise ``LaunchRefused`` with the reason."""
    if approval is None:
        raise LaunchRefused(
            f"no approval on record for gate {spec.gate}: the operator must reply "
            f'"approved" to /gate {spec.gate} first'
        )
    if approval.gate != spec.gate:
        raise LaunchRefused(
            f"the approval on hand is for gate {approval.gate}, but this job is gate {spec.gate}"
        )
    if not approval.approved:
        raise LaunchRefused(f"gate {spec.gate} is recorded but not approved")
    if spec.max_cost_inr > approval.max_inr:
        raise LaunchRefused(
            f"the spec ceiling of INR {spec.max_cost_inr:.0f} is above the gate "
            f"{spec.gate} authorisation of INR {approval.max_inr:.0f}"
        )

    # The gate's ceiling belongs to the gate, not to each spec in turn.
    committed = gate_exposure_inr(
        burn_rows, gate=spec.gate, inr_per_usd=inr_per_usd, excluding=spec.run_id
    )
    if committed + spec.max_cost_inr > approval.max_inr:
        raise LaunchRefused(
            f"gate {spec.gate} has INR {committed:,.0f} already committed by other "
            f"runs; adding this spec's INR {spec.max_cost_inr:,.0f} would reach "
            f"INR {committed + spec.max_cost_inr:,.0f} against an authorisation of "
            f"INR {approval.max_inr:,.0f}. Lower this spec's ceiling or re-gate."
        )

    rows = [row for row in burn_rows if row.run_id == spec.run_id]
    if not rows:
        raise LaunchRefused(
            f"no row for run_id {spec.run_id!r} in the burn tracker: write the "
            "`planned` row before launching"
        )
    row = rows[0]
    if row.gate != spec.gate:
        raise LaunchRefused(
            f"the burn tracker files {spec.run_id!r} under gate {row.gate}, "
            f"the spec says gate {spec.gate}"
        )
    if row.status != "planned":
        raise LaunchRefused(
            f"the burn tracker row for {spec.run_id!r} has status {row.status!r}, "
            "expected 'planned'"
        )

    # Gate exposure is summed from the tracker's ceilings, so the tracker is only a
    # sound basis for it if a spec cannot claim more than its own row says. Without
    # this, rows planned at INR 1 each could launch specs of INR 300 each and the
    # guard would count INR 1 apiece.
    if row.max_cost_inr <= 0:
        raise LaunchRefused(
            f"the burn tracker row for {spec.run_id!r} has no max_cost_inr. Exposure "
            f"against a gate is summed over its runs' ceilings, so a row without one "
            f"cannot authorise anything — add the ceiling to the `planned` row."
        )
    if spec.max_cost_inr > row.max_cost_inr:
        raise LaunchRefused(
            f"the spec ceiling of INR {spec.max_cost_inr:,.0f} is above the "
            f"INR {row.max_cost_inr:,.0f} its planned row commits. Raise the planned "
            f"row (and re-check the gate total) rather than the spec alone."
        )

    if balance_inr < spec.max_cost_inr:
        raise LaunchRefused(
            f"prepaid balance INR {balance_inr:.0f} is below the ceiling of "
            f"INR {spec.max_cost_inr:.0f}; the balance is the hard ceiling and only "
            "the operator tops it up"
        )

    if quoted_rate_usd_per_h > row.rate_usd_per_h * PRICE_TOLERANCE:
        raise LaunchRefused(
            f"quoted ${quoted_rate_usd_per_h:.3f}/h is more than 20% above the planned "
            f"${row.rate_usd_per_h:.3f}/h; re-quote and re-plan rather than absorb it"
        )

    estimated = spec.max_hours * quoted_rate_usd_per_h * inr_per_usd
    if estimated > spec.max_cost_inr:
        raise LaunchRefused(
            f"{spec.max_hours:g} h at ${quoted_rate_usd_per_h:.3f}/h is about "
            f"INR {estimated:.0f}, over the spec ceiling of INR {spec.max_cost_inr:.0f}"
        )

    alive = [p for p in running_pods if p.run_id == spec.run_id]
    if alive:
        raise LaunchRefused(
            f"run_id {spec.run_id!r} already has a live pod ({alive[0].pod_id}); "
            "pull and destroy it before relaunching"
        )

    return LaunchDecision(
        spec_run_id=spec.run_id,
        estimated_cost_inr=estimated,
        quoted_rate_usd_per_h=quoted_rate_usd_per_h,
        inr_per_usd=inr_per_usd,
    )


def pods_to_destroy(
    pods: Sequence[PodState],
    burn_rows: Sequence[BurnRow],
    *,
    now: datetime,
) -> list[tuple[PodState, str]]:
    """Pods the watchdog must destroy, each with the reason why.

    A pod the burn tracker has never heard of is destroyed on sight: it is either a
    leftover or a launch that bypassed the guards, and both cost money.
    """
    known = {row.run_id for row in burn_rows}
    doomed: list[tuple[PodState, str]] = []
    for p in pods:
        if p.run_id not in known:
            doomed.append((p, f"pod {p.pod_id} runs {p.run_id!r}, which is not in the burn tracker"))
        elif p.cost_inr > p.max_cost_inr:
            doomed.append(
                (
                    p,
                    f"pod {p.pod_id} has cost INR {p.cost_inr:.0f}, over its ceiling of "
                    f"INR {p.max_cost_inr:.0f}",
                )
            )
        elif p.deadline <= now:
            doomed.append(
                (p, f"pod {p.pod_id} is past its deadline ({p.deadline.isoformat()})")
            )
    return doomed
