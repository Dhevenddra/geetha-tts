"""Tests for ops.guards — the rules that stand between us and a surprise bill.

Runbook section 9: ``gpu.py launch`` refuses unless an approved gate exists, a
`planned` row exists in the burn tracker, the balance covers the ceiling, and the
quoted price is within 20% of the plan. ``status --enforce`` destroys any pod past
its deadline or unknown to the burn tracker.

Every rule here is a pure function so it can be tested without touching a provider
or spending anything.
"""
from datetime import datetime, timedelta, timezone

import pytest

from ops.guards import (
    BurnRow,
    GateApproval,
    LaunchRefused,
    PodState,
    check_launch,
    pods_to_destroy,
)
from ops.spec import JobSpec

SPEC = JobSpec.from_dict(
    {
        "run_id": "m1-pilot",
        "gate": "M1",
        "provider": "runpod",
        "gpu_type": "NVIDIA GeForce RTX 4090",
        "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1",
        "disk_gb": 100,
        "max_hours": 4.0,
        "max_cost_inr": 150.0,
        "command": "bash train.sh",
        "data_repos": ["Dhevenddra/geetha-tts-stagea-private"],
        "checkpoint_repo": "Dhevenddra/geetha-ckpt-private",
        "wandb_project": "geetha",
    }
)

APPROVED_M1 = GateApproval(gate="M1", approved=True, max_inr=300.0)
# max_cost_inr mirrors SPEC's ceiling: a planned row that commits less than the spec
# claims is refused, because gate exposure is summed from these ceilings.
PLANNED_ROW = BurnRow(
    run_id="m1-pilot", gate="M1", status="planned", planned_hours=4.0,
    rate_usd_per_h=0.34, cost_inr=0.0, max_cost_inr=300.0,
)


def launch(**overrides):
    kwargs = dict(
        spec=SPEC,
        approval=APPROVED_M1,
        burn_rows=[PLANNED_ROW],
        balance_inr=980.0,
        quoted_rate_usd_per_h=0.34,
        inr_per_usd=88.0,
        running_pods=[],
    )
    kwargs.update(overrides)
    return check_launch(**kwargs)


def test_a_fully_approved_launch_is_allowed_and_estimates_the_cost():
    decision = launch()
    # 4 h at $0.34 at 88 INR/USD = 119.68
    assert decision.estimated_cost_inr == pytest.approx(119.68, abs=0.01)


def test_no_approval_for_the_gate_refuses():
    with pytest.raises(LaunchRefused, match="no approval"):
        launch(approval=None)


def test_an_unapproved_gate_refuses():
    with pytest.raises(LaunchRefused, match="not approved"):
        launch(approval=GateApproval(gate="M1", approved=False, max_inr=300.0))


def test_an_approval_for_a_different_gate_refuses():
    with pytest.raises(LaunchRefused, match="gate"):
        launch(approval=GateApproval(gate="M2", approved=True, max_inr=1000.0))


def test_a_ceiling_above_what_the_gate_authorised_refuses():
    with pytest.raises(LaunchRefused, match="above the gate"):
        launch(approval=GateApproval(gate="M1", approved=True, max_inr=100.0))


def test_a_missing_burn_tracker_row_refuses():
    with pytest.raises(LaunchRefused, match="burn tracker"):
        launch(burn_rows=[])


def test_a_burn_row_that_is_not_planned_refuses():
    done = BurnRow(
        run_id="m1-pilot", gate="M1", status="done", planned_hours=4.0, rate_usd_per_h=0.34, cost_inr=120.0
    )
    with pytest.raises(LaunchRefused, match="planned"):
        launch(burn_rows=[done])


def test_a_burn_row_filed_under_another_gate_refuses():
    wrong = BurnRow(
        run_id="m1-pilot", gate="M2", status="planned", planned_hours=4.0, rate_usd_per_h=0.34, cost_inr=0.0
    )
    with pytest.raises(LaunchRefused, match="gate"):
        launch(burn_rows=[wrong])


def test_a_balance_below_the_ceiling_refuses():
    # The prepaid balance is the hard ceiling (the project rules). 100 < 150.
    with pytest.raises(LaunchRefused, match="balance"):
        launch(balance_inr=100.0)


def test_a_quoted_price_more_than_twenty_percent_over_plan_refuses():
    with pytest.raises(LaunchRefused, match="20%"):
        launch(quoted_rate_usd_per_h=0.34 * 1.21)


def test_a_quoted_price_exactly_twenty_percent_over_plan_is_allowed():
    assert launch(quoted_rate_usd_per_h=0.34 * 1.20) is not None


def test_a_quoted_price_below_plan_is_allowed():
    assert launch(quoted_rate_usd_per_h=0.20) is not None


def test_a_quote_whose_full_run_would_exceed_the_spec_ceiling_refuses():
    # Within +20% of plan per hour, but the hours make it blow the rupee ceiling.
    with pytest.raises(LaunchRefused, match="ceiling"):
        launch(quoted_rate_usd_per_h=0.34, inr_per_usd=200.0)


def test_launching_a_run_that_already_has_a_pod_refuses():
    existing = PodState(
        pod_id="abc", run_id="m1-pilot", hours_used=0.5, cost_inr=15.0,
        max_cost_inr=150.0, deadline=datetime(2026, 10, 30, 12, tzinfo=timezone.utc),
    )
    with pytest.raises(LaunchRefused, match="already"):
        launch(running_pods=[existing])


NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def pod(**overrides) -> PodState:
    kwargs = dict(
        pod_id="abc",
        run_id="m1-pilot",
        hours_used=1.0,
        cost_inr=30.0,
        max_cost_inr=150.0,
        deadline=NOW + timedelta(hours=3),
    )
    kwargs.update(overrides)
    return PodState(**kwargs)


def test_a_healthy_pod_inside_every_limit_is_left_alone():
    assert pods_to_destroy([pod()], [PLANNED_ROW], now=NOW) == []


def test_a_pod_past_its_deadline_is_destroyed():
    doomed = [p for p, _ in pods_to_destroy([pod(deadline=NOW - timedelta(minutes=1))], [PLANNED_ROW], now=NOW)]
    assert len(doomed) == 1
    reason = pods_to_destroy([pod(deadline=NOW - timedelta(minutes=1))], [PLANNED_ROW], now=NOW)[0][1]
    assert "deadline" in reason


def test_a_pod_unknown_to_the_burn_tracker_is_destroyed():
    result = pods_to_destroy([pod(run_id="mystery-run")], [PLANNED_ROW], now=NOW)
    assert len(result) == 1
    assert "burn tracker" in result[0][1]


def test_a_pod_over_its_rupee_ceiling_is_destroyed_even_before_its_deadline():
    # Time is not the only way to overspend: the hourly price can move.
    result = pods_to_destroy([pod(cost_inr=151.0)], [PLANNED_ROW], now=NOW)
    assert len(result) == 1
    assert "cost" in result[0][1]


def test_a_pod_at_eighty_percent_of_its_ceiling_is_flagged_but_not_destroyed():
    assert pods_to_destroy([pod(cost_inr=120.0)], [PLANNED_ROW], now=NOW) == []


def test_several_bad_pods_are_all_returned():
    bad = [pod(pod_id="a", run_id="mystery"), pod(pod_id="b", cost_inr=999.0)]
    assert len(pods_to_destroy(bad, [PLANNED_ROW], now=NOW)) == 2


# --- a gate's ceiling applies to the gate, not to one spec at a time -------------
#
# Flagged by security review, 27 Oct. check_launch compared spec.max_cost_inr against
# approval.max_inr, so three specs of INR 300 each under a gate approved for INR 300
# all passed individually and spent INR 900 against a INR 300 authorisation. A gate can
# legitimately carry several run_ids (M4 has the ASR runs), so this is reachable by
# planning the work the runbook already describes, not by anything exotic.

from ops.guards import gate_exposure_inr  # noqa: E402


def _spec(run_id="m1-pilot", gate="M1", max_cost_inr=300.0, max_hours=2.0):
    class _S:
        pass

    spec = _S()
    spec.run_id = run_id
    spec.gate = gate
    spec.max_cost_inr = max_cost_inr
    spec.max_hours = max_hours
    spec.gpu_type = "NVIDIA GeForce RTX 4090"
    return spec


def _row(run_id, gate="M1", status="planned", hours=2.0, rate=0.34, cost=0.0,
         ceiling=300.0):
    return BurnRow(run_id=run_id, gate=gate, status=status, planned_hours=hours,
                   rate_usd_per_h=rate, cost_inr=cost, max_cost_inr=ceiling)


def _approval(gate="M1", max_inr=300.0):
    return GateApproval(gate=gate, approved=True, max_inr=max_inr)


def _check(spec, rows, approval, balance=100_000.0, rate=0.34):
    return check_launch(
        spec=spec, approval=approval, burn_rows=rows, balance_inr=balance,
        quoted_rate_usd_per_h=rate, inr_per_usd=88.0, running_pods=[],
    )


def test_a_second_run_under_a_gate_cannot_spend_the_whole_ceiling_again():
    # m0-probe already committed ~INR 60 of M1; a INR 300 spec no longer fits.
    rows = [
        _row("m0-selfdelete-probe", hours=0.25, ceiling=15.0, cost=0.0),
        _row("m1-pilot", hours=2.0),
    ]
    with pytest.raises(LaunchRefused, match="gate M1"):
        _check(_spec(max_cost_inr=300.0), rows, _approval(max_inr=300.0))


def test_a_run_that_fits_what_is_left_of_the_gate_is_allowed():
    rows = [
        _row("m0-selfdelete-probe", hours=0.25, ceiling=15.0),  # the real probe ceiling
        _row("m1-pilot", hours=2.0),
    ]
    decision = _check(_spec(max_cost_inr=280.0), rows, _approval(max_inr=300.0))
    assert decision.spec_run_id == "m1-pilot"


def test_a_run_under_a_different_gate_does_not_consume_this_gates_ceiling():
    rows = [
        _row("m2-stage-a", gate="M2", hours=10.0),
        _row("m1-pilot", gate="M1", hours=2.0),
    ]
    assert _check(_spec(max_cost_inr=300.0), rows, _approval(max_inr=300.0))


def test_the_specs_own_row_is_not_counted_against_it_twice():
    # The spec's planned row is the one being launched; counting it as prior exposure
    # would refuse every first launch.
    rows = [_row("m1-pilot", hours=2.0)]
    assert _check(_spec(max_cost_inr=300.0), rows, _approval(max_inr=300.0))


def test_a_finished_run_consumes_its_actual_cost_not_its_plan():
    rows = [
        _row("m0-selfdelete-probe", status="done", hours=0.25, ceiling=15.0, cost=8.0),
        _row("m1-pilot", hours=2.0),
    ]
    assert _check(_spec(max_cost_inr=290.0), rows, _approval(max_inr=300.0))


def test_a_running_run_that_has_overshot_its_plan_consumes_the_larger_number():
    rows = [
        _row("m2-first", status="running", hours=1.0, rate=0.34, cost=250.0),
        _row("m2-second", hours=1.0),
    ]
    with pytest.raises(LaunchRefused, match="gate M2"):
        _check(_spec(run_id="m2-second", gate="M2", max_cost_inr=100.0),
               rows, _approval(gate="M2", max_inr=300.0))


def test_gate_exposure_ignores_the_run_being_launched():
    rows = [_row("a", hours=1.0), _row("b", hours=1.0)]
    both = gate_exposure_inr(rows, gate="M1", inr_per_usd=88.0, excluding="")
    one = gate_exposure_inr(rows, gate="M1", inr_per_usd=88.0, excluding="a")
    assert one == pytest.approx(both / 2)


# --- one planned row authorises one launch ---------------------------------------


def test_a_row_already_marked_running_cannot_be_launched_again():
    # Second half of the same finding: destroy forgot the registry row but left the
    # burn row on `planned`, so one authorisation permitted unlimited relaunches.
    # gpu.py now moves the row to `running` at launch and `done` at destroy.
    rows = [_row("m1-pilot", status="running", hours=2.0)]
    with pytest.raises(LaunchRefused, match="'running'"):
        _check(_spec(), rows, _approval())


def test_a_row_already_marked_done_cannot_be_launched_again():
    rows = [_row("m1-pilot", status="done", hours=2.0, cost=120.0)]
    with pytest.raises(LaunchRefused, match="'done'"):
        _check(_spec(), rows, _approval())


# --- the spec must match the row that authorised it ------------------------------
#
# Second security review, 27 Oct. Gate exposure is computed from the burn tracker's
# max_cost_inr, but nothing tied that figure to the ceiling of the spec actually
# launched. Planning five runs at a ceiling of INR 1 each and then launching five specs
# each claiming INR 300 committed INR 1,500 of an M4 approved for INR 1,300, while the
# guard counted INR 4. The tracker is only a sound basis for exposure if a spec cannot
# claim more than its own row says.


def test_a_spec_claiming_more_than_its_planned_row_is_refused():
    rows = [_row("m1-pilot", hours=0.01, ceiling=1.0)]
    with pytest.raises(LaunchRefused, match="planned row"):
        _check(_spec(max_cost_inr=300.0), rows, _approval(max_inr=300.0))


def test_a_spec_claiming_exactly_its_planned_row_is_allowed():
    rows = [_row("m1-pilot", hours=2.0, ceiling=300.0)]
    assert _check(_spec(max_cost_inr=300.0), rows, _approval(max_inr=300.0))


def test_a_spec_claiming_less_than_its_planned_row_is_allowed():
    # Lowering a ceiling at launch is safe; raising it is the problem.
    rows = [_row("m1-pilot", hours=2.0, ceiling=300.0)]
    assert _check(_spec(max_cost_inr=100.0), rows, _approval(max_inr=300.0))


def test_a_planned_row_with_no_ceiling_cannot_authorise_anything():
    # A row written before the column existed, or by hand without it. Reading it as
    # "no ceiling" would make an unspecified row the most permissive kind.
    rows = [_row("m1-pilot", hours=2.0, ceiling=0.0)]
    with pytest.raises(LaunchRefused, match="max_cost_inr"):
        _check(_spec(max_cost_inr=300.0), rows, _approval(max_inr=300.0))


def test_the_five_small_rows_bypass_is_closed():
    # The exact scenario measured above: five rows planned at INR 1, five specs at 300.
    rows = [_row(f"r{index}", gate="M4", hours=0.01, ceiling=1.0) for index in range(5)]
    approval = _approval(gate="M4", max_inr=1300.0)
    for index in range(5):
        with pytest.raises(LaunchRefused):
            _check(_spec(run_id=f"r{index}", gate="M4", max_cost_inr=300.0), rows, approval)
