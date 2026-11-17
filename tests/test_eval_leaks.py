"""Tests for geetha.eval.leaks — what to do when an eval prompt appears in training.

Runbook rule 3 is explicit about the remedy: "any sentence (normalised hash) in a
held-out split is **removed from every training manifest**". So a collision is not a
reason to rewrite the eval set — it is a removal instruction for the ETL.

That matters for the intent probes. A probe has to be a short, common phrase ("சரி",
"போ") or its intent cannot flip with delivery, so collisions are inherent to the design.
Measured 26 Oct 2026: honouring rule 3 for all 16 colliding probes costs 2,044 clips out
of 909,265 training rows — 0.22%. Negligible, so the rule stands as written.

The guard is on cost, not on collisions: if removal would cost a large share of training
data, the prompt set is badly designed and that should stop the build.
"""
import pytest

from geetha.eval.leaks import LeakRemovalPlan, MAX_REMOVAL_FRACTION, plan_leak_removal


def test_a_plan_lists_every_colliding_text_with_its_clip_cost():
    plan = plan_leak_removal(
        leaks=[("p02", "சரி."), ("p06", "போ.")],
        clip_counts={"p02": 1341, "p06": 371},
        training_rows=909_265,
    )
    assert plan.rows == [("p02", "சரி.", 1341), ("p06", "போ.", 371)]
    assert plan.clips_removed == 1712


def test_the_plan_reports_the_fraction_of_training_data_lost():
    plan = plan_leak_removal(
        leaks=[("p02", "சரி.")], clip_counts={"p02": 2044}, training_rows=909_265
    )
    assert plan.fraction == pytest.approx(2044 / 909_265)
    assert plan.fraction < MAX_REMOVAL_FRACTION


def test_no_collisions_gives_an_empty_plan_that_is_still_valid():
    plan = plan_leak_removal(leaks=[], clip_counts={}, training_rows=1000)
    assert plan.rows == []
    assert plan.clips_removed == 0
    assert plan.fraction == 0.0


def test_a_removal_cost_above_the_guard_is_refused():
    # Removing a large share of training data means the prompt set is wrong, not that
    # the rule is wrong.
    with pytest.raises(ValueError, match="2.0%"):
        plan_leak_removal(
            leaks=[("x", "common phrase")], clip_counts={"x": 20}, training_rows=1000
        )


def test_the_guard_sits_at_one_percent():
    assert MAX_REMOVAL_FRACTION == 0.01


def test_a_leak_with_no_recorded_clip_count_is_refused():
    # Writing a removal instruction without knowing its cost would hide the cost.
    with pytest.raises(ValueError, match="no clip count"):
        plan_leak_removal(leaks=[("p02", "சரி.")], clip_counts={}, training_rows=1000)


def test_the_plan_is_ordered_by_cost_so_the_expensive_ones_are_obvious():
    plan = plan_leak_removal(
        leaks=[("a", "x"), ("b", "y"), ("c", "z")],
        clip_counts={"a": 1, "b": 500, "c": 50},
        training_rows=909_265,
    )
    assert [row[0] for row in plan.rows] == ["b", "c", "a"]


def test_plan_rows_can_be_written_as_csv_fields():
    plan = plan_leak_removal(leaks=[("p02", "சரி.")], clip_counts={"p02": 3}, training_rows=1000)
    assert isinstance(plan, LeakRemovalPlan)
    assert plan.rows[0] == ("p02", "சரி.", 3)
