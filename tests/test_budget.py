"""When a paid training run must stop (the pod trainer's stop rules).

the project rules: "If a job misbehaves (loss NaN, probe CER rising, idle GPU), stop it, pull
what exists, and tell the operator... A failed run is fine; a silently burning pod is
not." These are those rules, as code, so the decision is not left to whoever is watching
the logs — which on a 25-hour Stage A run is nobody for most of it.

Divergence is included deliberately. A NaN is obvious and a wall-clock budget is
trivial; a loss that has quietly climbed to three times where it started burns the whole
budget and produces a worse model than the base, with a log full of numbers that each
look plausible on their own.
"""

from __future__ import annotations

import math

import pytest

from geetha.train.budget import TrainingBudget


def budget(**overrides) -> TrainingBudget:
    spec = dict(max_seconds=3600.0, max_steps=1000, divergence_factor=3.0,
                divergence_after=50)
    spec.update(overrides)
    return TrainingBudget(**spec)


class TestHealthyRun:
    def test_a_healthy_run_is_not_stopped(self):
        assert budget().should_stop(step=10, elapsed=100.0, loss=4.2) is None

    def test_a_falling_loss_is_never_divergence(self):
        guard = budget()
        for step in range(200):
            assert guard.should_stop(step=step, elapsed=1.0, loss=5.0 - step * 0.01) is None


class TestStopReasons:
    def test_the_wall_clock_budget_stops_the_run(self):
        assert budget().should_stop(step=1, elapsed=3600.0, loss=4.0) == "budget_exhausted"

    def test_the_step_cap_stops_the_run(self):
        assert budget().should_stop(step=1000, elapsed=1.0, loss=4.0) == "max_steps"

    def test_a_nan_loss_stops_the_run(self):
        assert budget().should_stop(step=5, elapsed=1.0, loss=float("nan")) == "nan_loss"

    def test_an_infinite_loss_stops_the_run(self):
        assert budget().should_stop(step=5, elapsed=1.0, loss=math.inf) == "nan_loss"

    def test_a_nan_outranks_the_budget_so_the_report_names_the_real_problem(self):
        assert budget().should_stop(step=5, elapsed=99_999.0, loss=float("nan")) == "nan_loss"


class TestDivergence:
    def test_a_loss_climbing_past_the_factor_stops_the_run(self):
        guard = budget()
        for step in range(60):
            guard.should_stop(step=step, elapsed=1.0, loss=4.0)
        assert guard.should_stop(step=60, elapsed=1.0, loss=13.0) == "diverging"

    def test_divergence_is_not_judged_before_a_baseline_exists(self):
        # Early loss is noisy; judging it would stop healthy runs in their first minute.
        guard = budget(divergence_after=50)
        guard.should_stop(step=0, elapsed=1.0, loss=4.0)
        assert guard.should_stop(step=1, elapsed=1.0, loss=400.0) is None

    def test_the_baseline_is_the_opening_window_not_the_last_step(self):
        # A single bad step must not reset what "normal" means, or a spike would
        # silently raise the bar for every step after it.
        guard = budget(divergence_after=10)
        for step in range(10):
            guard.should_stop(step=step, elapsed=1.0, loss=2.0)
        guard.should_stop(step=10, elapsed=1.0, loss=5.0)
        assert guard.should_stop(step=11, elapsed=1.0, loss=6.1) == "diverging"

    def test_a_loss_just_under_the_factor_is_tolerated(self):
        guard = budget(divergence_after=10, divergence_factor=3.0)
        for step in range(10):
            guard.should_stop(step=step, elapsed=1.0, loss=2.0)
        assert guard.should_stop(step=10, elapsed=1.0, loss=5.9) is None

    def test_divergence_can_be_switched_off(self):
        guard = budget(divergence_factor=None, divergence_after=5)
        for step in range(10):
            guard.should_stop(step=step, elapsed=1.0, loss=1.0)
        assert guard.should_stop(step=11, elapsed=1.0, loss=10_000.0) is None


class TestRejectsNonsense:
    def test_a_non_positive_time_budget_is_refused(self):
        with pytest.raises(ValueError, match="max_seconds"):
            budget(max_seconds=0.0)

    def test_a_non_positive_step_cap_is_refused(self):
        with pytest.raises(ValueError, match="max_steps"):
            budget(max_steps=0)

    def test_a_divergence_factor_at_or_below_one_is_refused(self):
        # A factor of 1.0 would stop the run the first time the loss ticked up.
        with pytest.raises(ValueError, match="divergence_factor"):
            budget(divergence_factor=1.0)


class TestReporting:
    def test_the_budget_reports_the_baseline_it_measured(self):
        guard = budget(divergence_after=4)
        for step in range(4):
            guard.should_stop(step=step, elapsed=1.0, loss=3.0)
        assert guard.baseline_loss == pytest.approx(3.0)

    def test_the_baseline_is_unset_until_the_window_closes(self):
        guard = budget(divergence_after=4)
        guard.should_stop(step=0, elapsed=1.0, loss=3.0)
        assert guard.baseline_loss is None
