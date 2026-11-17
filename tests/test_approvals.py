"""Tests for ops.approvals — the machine-readable record of what the operator allowed.

``check_launch`` has always taken a ``GateApproval``; nothing could produce one, so
every launch refused. This is the loader, and its job is to be *harder* to get past
than prose in DECISIONS.md, not easier: the ceiling in a row is cross-checked against
the gate budget the plan authorised, so a mis-recorded approval cannot raise its own
limit.
"""
import pytest

from ops.approvals import (
    APPROVAL_COLUMNS,
    GATE_BUDGETS,
    ApprovalError,
    read_approvals,
    total_approved_inr,
)

HEADER = ",".join(APPROVAL_COLUMNS)


def _write(tmp_path, *rows):
    path = tmp_path / "gate_approvals.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_an_approved_gate_loads_with_its_ceiling(tmp_path):
    path = _write(tmp_path, "2026-10-30,M1,300,operator,approved in chat")
    approvals = read_approvals(path)
    assert approvals["M1"].approved is True
    assert approvals["M1"].max_inr == 300.0
    assert approvals["M1"].gate == "M1"


def test_a_missing_file_means_no_approvals_not_an_error(tmp_path):
    # Nothing is approved yet is the normal state of this project, and it must read as
    # "refuse every launch", not as a crash that someone is tempted to work around.
    assert read_approvals(tmp_path / "absent.csv") == {}


def test_a_ceiling_above_the_gate_budget_is_refused(tmp_path):
    # D-013 sets M1 at INR 300. A row claiming more is either a typo or a drift from
    # the plan; either way it must not authorise the extra.
    path = _write(tmp_path, "2026-10-30,M1,900,operator,oops")
    with pytest.raises(ApprovalError, match="above the M1 budget"):
        read_approvals(path)


def test_a_ceiling_at_exactly_the_gate_budget_is_allowed(tmp_path):
    path = _write(tmp_path, f"2026-10-30,M2,{GATE_BUDGETS['M2']:.0f},operator,")
    assert read_approvals(path)["M2"].max_inr == GATE_BUDGETS["M2"]


def test_an_unknown_gate_is_refused(tmp_path):
    path = _write(tmp_path, "2026-10-30,M9,100,operator,")
    with pytest.raises(ApprovalError, match="M9"):
        read_approvals(path)


def test_an_approval_not_from_the_operator_is_refused(tmp_path):
    # Only one person on this project approves money. A row signed by anything else is
    # the failure this file exists to prevent.
    path = _write(tmp_path, "2026-10-30,M1,300,assistant,I decided it was fine")
    with pytest.raises(ApprovalError, match="operator"):
        read_approvals(path)


def test_two_rows_for_one_gate_are_refused(tmp_path):
    # Not "last wins": two rows mean nobody knows which ceiling is in force.
    path = _write(
        tmp_path,
        "2026-10-30,M1,300,operator,",
        "2026-10-31,M1,250,operator,",
    )
    with pytest.raises(ApprovalError, match="duplicate"):
        read_approvals(path)


def test_a_negative_or_zero_ceiling_is_refused(tmp_path):
    with pytest.raises(ApprovalError, match="positive"):
        read_approvals(_write(tmp_path, "2026-10-30,M1,0,operator,"))


def test_a_non_numeric_ceiling_is_refused_rather_than_read_as_zero(tmp_path):
    with pytest.raises(ApprovalError, match="max_inr"):
        read_approvals(_write(tmp_path, "2026-10-30,M1,three hundred,operator,"))


def test_a_column_mismatch_is_refused(tmp_path):
    path = tmp_path / "gate_approvals.csv"
    path.write_text("gate,max_inr\nM1,300\n", encoding="utf-8")
    with pytest.raises(ApprovalError, match="column"):
        read_approvals(path)


def test_a_missing_date_is_refused(tmp_path):
    # An approval with no date cannot be checked against the sprint, and an undated
    # authorisation is the kind that quietly stays valid forever.
    with pytest.raises(ApprovalError, match="date"):
        read_approvals(_write(tmp_path, ",M1,300,operator,"))


def test_a_malformed_date_is_refused(tmp_path):
    with pytest.raises(ApprovalError, match="date"):
        read_approvals(_write(tmp_path, "30-10-2026,M1,300,operator,"))


def test_total_approved_is_the_sum_of_the_ceilings(tmp_path):
    path = _write(
        tmp_path,
        "2026-10-30,M1,300,operator,",
        "2026-10-31,M2,1000,operator,",
    )
    assert total_approved_inr(read_approvals(path)) == 1300.0


def test_the_gate_budgets_match_the_plan():
    # D-013: M1 300, M2 1,000, M3 500, M4 1,300, M5 600, M6 600. That is INR 4,300 of
    # gates plus the 1,100 reserve = 5,400 against the 5,700 cap, and the whole of it
    # sits above the 4,600 stop rule for Mon 9 Nov — which is why the stop rule is
    # enforced on *spend* in the burn tracker, not on what is approved here.
    assert GATE_BUDGETS == {
        "M1": 300.0, "M2": 1000.0, "M3": 500.0,
        "M4": 1300.0, "M5": 600.0, "M6": 600.0,
    }
    assert sum(GATE_BUDGETS.values()) == 4300.0
