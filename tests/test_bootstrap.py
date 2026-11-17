"""Tests for geetha.scoring.bootstrap.

Runbook section 10: every published number carries a 95% bootstrap CI. Comparisons
between two systems are *paired* -- both systems are resampled on the same
utterances, because they were measured on the same utterances.
"""
import pytest

from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap
from geetha.scoring.metrics import ErrorCounts


def counts(errors: int, length: int) -> ErrorCounts:
    return ErrorCounts(substitutions=errors, deletions=0, insertions=0, reference_length=length)


def test_the_point_estimate_is_the_corpus_rate_not_a_bootstrap_mean():
    result = bootstrap_ci([counts(1, 10), counts(3, 10)], n_resamples=500, seed=1234)
    assert result.point == pytest.approx(4 / 20)


def test_the_interval_brackets_the_point_estimate():
    result = bootstrap_ci([counts(e, 10) for e in range(10)], n_resamples=500, seed=1234)
    assert result.low <= result.point <= result.high


def test_a_higher_confidence_level_gives_a_wider_interval():
    data = [counts(e % 5, 10) for e in range(40)]
    narrow = bootstrap_ci(data, n_resamples=2000, seed=1234, confidence=0.90)
    wide = bootstrap_ci(data, n_resamples=2000, seed=1234, confidence=0.99)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_the_same_seed_reproduces_the_same_interval():
    data = [counts(e % 5, 10) for e in range(40)]
    first = bootstrap_ci(data, n_resamples=1000, seed=1234)
    second = bootstrap_ci(data, n_resamples=1000, seed=1234)
    assert (first.low, first.high) == (second.low, second.high)


def test_a_different_seed_gives_a_different_interval():
    data = [counts(e % 5, 10) for e in range(40)]
    first = bootstrap_ci(data, n_resamples=1000, seed=1234)
    second = bootstrap_ci(data, n_resamples=1000, seed=99)
    assert (first.low, first.high) != (second.low, second.high)


def test_bootstrapping_an_empty_corpus_raises():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], n_resamples=100, seed=1234)


def test_two_identical_systems_show_no_difference():
    data = [counts(e % 4, 10) for e in range(30)]
    result = paired_bootstrap(data, data, n_resamples=500, seed=1234)
    assert result.delta == 0.0
    assert result.low == 0.0 and result.high == 0.0
    assert result.p_value == 1.0


def test_a_system_better_on_every_utterance_gives_a_negative_interval():
    better = [counts(1, 10) for _ in range(30)]
    worse = [counts(5, 10) for _ in range(30)]
    result = paired_bootstrap(better, worse, n_resamples=2000, seed=1234)
    assert result.delta == pytest.approx(-0.4)
    assert result.high < 0
    assert result.p_value < 0.01


def test_pairing_requires_the_same_utterances_on_both_sides():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([counts(1, 10)], [counts(1, 10), counts(2, 10)], n_resamples=100, seed=1234)


def test_a_tiny_difference_on_noisy_data_is_not_called_significant():
    # Each system wins on some utterances and loses on others, with almost nothing
    # between them overall. That must come back as "no measurable difference".
    a = [counts(e % 5, 10) for e in range(30)]
    b = [counts((e + 1) % 5, 10) for e in range(30)]
    result = paired_bootstrap(a, b, n_resamples=2000, seed=1234)
    assert result.low < 0 < result.high
    assert result.p_value > 0.05


def test_a_one_sided_difference_gives_an_interval_that_stops_at_zero():
    # System b is worse on exactly one utterance and identical elsewhere, so no
    # resample can make a look worse: the upper bound is zero, not positive.
    a = [counts(e % 4, 10) for e in range(30)]
    b = list(a)
    b[0] = counts(a[0].errors + 1, 10)
    result = paired_bootstrap(a, b, n_resamples=2000, seed=1234)
    assert result.delta < 0
    assert result.low < 0
    assert result.high == 0.0


class TestTwoSampleCI:
    """Unpaired bootstrap — the contamination check's arms are different clips."""

    def test_a_clear_separation_excludes_zero(self):
        from geetha.scoring.bootstrap import two_sample_ci

        result = two_sample_ci([10.0] * 50, [2.0] * 50)
        assert result.delta == pytest.approx(8.0)
        assert result.low > 0.0

    def test_identical_samples_straddle_zero(self):
        from geetha.scoring.bootstrap import two_sample_ci

        values = [float(i % 7) for i in range(60)]
        result = two_sample_ci(values, list(values), seed=99)
        assert result.low <= 0.0 <= result.high
        assert result.p_value > 0.05

    def test_arms_need_not_be_the_same_size(self):
        from geetha.scoring.bootstrap import two_sample_ci

        result = two_sample_ci([5.0] * 50, [1.0] * 137)
        assert result.delta == pytest.approx(4.0)

    def test_it_is_reproducible_from_the_seed(self):
        from geetha.scoring.bootstrap import two_sample_ci

        a = [float(i % 11) for i in range(40)]
        b = [float(i % 5) for i in range(40)]
        first = two_sample_ci(a, b, seed=2026)
        second = two_sample_ci(a, b, seed=2026)
        assert (first.low, first.high) == (second.low, second.high)

    def test_an_empty_sample_is_refused(self):
        from geetha.scoring.bootstrap import two_sample_ci

        with pytest.raises(ValueError):
            two_sample_ci([], [1.0, 2.0])

    def test_an_unknown_statistic_is_refused(self):
        from geetha.scoring.bootstrap import two_sample_ci

        with pytest.raises(ValueError):
            two_sample_ci([1.0], [2.0], statistic="mode")
