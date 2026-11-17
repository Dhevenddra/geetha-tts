"""Tests for geetha.audio.events — turning detector scores into an event count.

This is the arithmetic the release headline rests on. "Geetha breathes 14 times a
minute, the base model zero" is this function applied to an audio tagger's output, so a
bug here inflates the claim rather than breaking it, which is the worse direction.

The hard part is not thresholding, it is deciding what counts as *one* event: a breath
spanning three analysis windows is one breath, and a single stray window above
threshold is noise, not a breath.
"""
import pytest

from geetha.audio.events import (
    DEFAULT_MIN_EVENT_SECONDS,
    DEFAULT_MIN_GAP_SECONDS,
    EventSpan,
    count_events,
    events_per_minute,
    rate_with_ci,
)


def _scores(pattern: str, hop: float = 0.5, high: float = 0.9, low: float = 0.1):
    """'..##..' -> a score per window, '#' above threshold."""
    return [(index * hop, high if char == "#" else low) for index, char in enumerate(pattern)]


# --- what counts as one event ----------------------------------------------------


def test_no_window_above_threshold_is_no_events():
    assert count_events(_scores("......"), threshold=0.5, hop_seconds=0.5) == []


def test_one_run_of_windows_is_one_event():
    # A breath lasting 1.5 s spans three 0.5 s windows. It is one breath.
    spans = count_events(_scores("..###."), threshold=0.5, hop_seconds=0.5,
                         min_event_seconds=0.5)
    assert len(spans) == 1
    assert spans[0] == EventSpan(start=1.0, end=2.5)


def test_two_runs_separated_by_a_clear_gap_are_two_events():
    spans = count_events(_scores("##....##"), threshold=0.5, hop_seconds=0.5,
                         min_event_seconds=0.5, min_gap_seconds=0.5)
    assert len(spans) == 2


def test_two_runs_closer_than_the_minimum_gap_are_merged():
    # One breath that dipped below threshold for a single window is still one breath.
    spans = count_events(_scores("##.##"), threshold=0.5, hop_seconds=0.5,
                         min_event_seconds=0.5, min_gap_seconds=1.0)
    assert len(spans) == 1
    assert spans[0].end == pytest.approx(2.5)


def test_a_single_stray_window_is_discarded_as_too_short():
    # Without this, detector jitter becomes breaths and the headline inflates.
    assert count_events(_scores("...#..."), threshold=0.5, hop_seconds=0.5,
                        min_event_seconds=0.6) == []


def test_a_run_exactly_at_the_minimum_duration_is_kept():
    spans = count_events(_scores("..##.."), threshold=0.5, hop_seconds=0.5,
                         min_event_seconds=1.0)
    assert len(spans) == 1


def test_an_event_running_to_the_end_of_the_clip_is_closed_not_dropped():
    spans = count_events(_scores("....##"), threshold=0.5, hop_seconds=0.5,
                         min_event_seconds=0.5)
    assert len(spans) == 1
    assert spans[0].end == pytest.approx(3.0)


def test_the_threshold_is_exclusive_on_equality_being_below():
    # A score exactly at the threshold counts as above it, so a threshold swept over
    # the observed scores cannot skip a value.
    spans = count_events([(0.0, 0.5), (0.5, 0.5)], threshold=0.5, hop_seconds=0.5,
                         min_event_seconds=0.5)
    assert len(spans) == 1


def test_defaults_are_stated_rather_than_hidden():
    # These two numbers decide the headline count, so they are named constants a report
    # can quote, not literals buried in a loop.
    assert DEFAULT_MIN_EVENT_SECONDS > 0
    assert DEFAULT_MIN_GAP_SECONDS > 0


def test_empty_scores_are_no_events_not_an_error():
    assert count_events([], threshold=0.5, hop_seconds=0.5) == []


def test_a_non_positive_hop_is_rejected():
    with pytest.raises(ValueError):
        count_events(_scores("##"), threshold=0.5, hop_seconds=0.0)


# --- the rate --------------------------------------------------------------------


def test_events_per_minute_is_events_over_duration():
    assert events_per_minute(3, 60.0) == 3.0
    assert events_per_minute(3, 30.0) == 6.0


def test_a_system_that_emits_nothing_rates_zero():
    # The number C2 predicts for the base model. It must be 0.0, not None.
    assert events_per_minute(0, 42.0) == 0.0


def test_events_per_minute_rejects_zero_duration():
    with pytest.raises(ValueError):
        events_per_minute(1, 0.0)


# --- the published figure carries an interval ------------------------------------


def test_the_rate_comes_with_a_confidence_interval():
    # Runbook rule: every published number has a 95% CI. A per-clip bootstrap over
    # (events, seconds) pairs is what makes "14.2 [12.1, 16.4]" possible.
    clips = [(3, 10.0)] * 40
    result = rate_with_ci(clips, seed=1)
    assert result.point == pytest.approx(18.0)
    assert result.low <= result.point <= result.high


def test_a_zero_rate_still_gets_an_interval_and_does_not_divide_by_zero():
    result = rate_with_ci([(0, 10.0)] * 20, seed=1)
    assert result.point == 0.0
    assert result.low == 0.0
    assert result.high == 0.0


def test_the_interval_narrows_with_more_clips():
    few = rate_with_ci([(2, 10.0), (0, 10.0), (4, 10.0)] * 3, seed=7)
    many = rate_with_ci([(2, 10.0), (0, 10.0), (4, 10.0)] * 60, seed=7)
    assert (many.high - many.low) < (few.high - few.low)


def test_rate_with_ci_rejects_an_empty_set():
    with pytest.raises(ValueError):
        rate_with_ci([], seed=1)


def test_rate_with_ci_is_deterministic_for_a_seed():
    clips = [(1, 5.0), (0, 5.0), (3, 5.0)] * 10
    assert rate_with_ci(clips, seed=11) == rate_with_ci(clips, seed=11)
