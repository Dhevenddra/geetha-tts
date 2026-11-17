"""Tests for geetha.audio.join — reassembling chunked synthesis into one clip."""
import numpy as np
import pytest

from geetha.audio.join import join_clips


def test_a_single_clip_is_returned_unchanged():
    wave = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    joined = join_clips([wave], sample_rate=24_000, gap_seconds=0.1)
    assert np.array_equal(joined, wave)


def test_two_clips_are_separated_by_exactly_one_gap():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(50, dtype=np.float32)
    joined = join_clips([a, b], sample_rate=1_000, gap_seconds=0.1)
    assert len(joined) == 100 + 100 + 50  # 0.1 s at 1 kHz = 100 samples


def test_no_gap_is_added_before_the_first_or_after_the_last_clip():
    # Padding the ends would inflate every duration we measure.
    a = np.full(10, 0.5, dtype=np.float32)
    joined = join_clips([a, a], sample_rate=1_000, gap_seconds=0.01)
    assert joined[0] == 0.5
    assert joined[-1] == 0.5


def test_the_gap_is_silent():
    a = np.ones(10, dtype=np.float32)
    joined = join_clips([a, a], sample_rate=1_000, gap_seconds=0.01)
    assert np.all(joined[10:20] == 0.0)


def test_three_clips_get_two_gaps():
    a = np.ones(10, dtype=np.float32)
    joined = join_clips([a, a, a], sample_rate=1_000, gap_seconds=0.01)
    assert len(joined) == 30 + 2 * 10


def test_a_zero_gap_is_plain_concatenation():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    joined = join_clips([a, b], sample_rate=24_000, gap_seconds=0.0)
    assert np.array_equal(joined, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_the_output_stays_float32():
    # soundfile writes what it is given; a silent promotion to float64 would double
    # every archived clip.
    a = np.ones(4, dtype=np.float32)
    assert join_clips([a, a], sample_rate=1_000, gap_seconds=0.01).dtype == np.float32


def test_an_empty_list_is_rejected():
    with pytest.raises(ValueError):
        join_clips([], sample_rate=24_000, gap_seconds=0.1)


def test_a_negative_gap_is_rejected():
    a = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError):
        join_clips([a, a], sample_rate=24_000, gap_seconds=-0.01)


def test_a_non_positive_sample_rate_is_rejected():
    a = np.ones(4, dtype=np.float32)
    with pytest.raises(ValueError):
        join_clips([a, a], sample_rate=0, gap_seconds=0.1)
