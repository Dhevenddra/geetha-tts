"""The loudness gate: a mute clip must never reach a judge.

The fixtures at the bottom are the pilot's own clips. The operator found the silent ones
by ear on 6 Nov; this gate has to find the same three without him.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from geetha.eval.validity import SILENCE_DBFS, check_clip, rms_dbfs, summarise

RATE = 24_000


def tone(seconds: float, dbfs: float) -> np.ndarray:
    """A sine whose RMS is exactly ``dbfs``."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (10 ** (dbfs / 20) * math.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestLevel:
    def test_rms_of_a_known_tone(self):
        assert rms_dbfs(tone(1.0, -26.0)) == pytest.approx(-26.0, abs=0.05)

    def test_digital_silence_and_an_empty_clip_are_minus_infinity(self):
        assert rms_dbfs(np.zeros(100)) == -math.inf
        assert rms_dbfs([]) == -math.inf


class TestCheckClip:
    def test_ordinary_speech_level_is_valid(self):
        check = check_clip(tone(5.0, -28.0), RATE, cap_seconds=21.93)
        assert check.valid and check.reason == ""

    @pytest.mark.parametrize("level", [-58.3, -53.4, -49.0])
    def test_the_pilots_three_silent_levels_are_invalid(self, level):
        check = check_clip(tone(3.0, level), RATE)
        assert not check.valid and check.silent and "silent" in check.reason

    def test_the_quietest_real_pilot_clip_is_well_clear_of_the_threshold(self):
        assert check_clip(tone(3.0, -31.0), RATE).valid
        assert -31.0 - SILENCE_DBFS >= 14.0

    def test_an_empty_clip_is_invalid_rather_than_an_error(self):
        assert not check_clip(np.zeros(0, dtype=np.float32), RATE).valid

    def test_a_loud_clip_at_the_token_cap_is_still_invalid(self):
        check = check_clip(tone(21.93, -27.0), RATE, cap_seconds=21.93)
        assert not check.valid and check.hit_cap and not check.silent

    def test_a_clip_just_short_of_the_cap_counts_as_hitting_it(self):
        assert check_clip(tone(21.8, -27.0), RATE, cap_seconds=21.93).hit_cap
        assert not check_clip(tone(21.0, -27.0), RATE, cap_seconds=21.93).hit_cap

    def test_direct_evidence_of_self_termination_outranks_the_length(self):
        long_but_ended = check_clip(tone(21.93, -27.0), RATE, cap_seconds=21.93,
                                    ended_itself=True)
        short_but_cut = check_clip(tone(4.0, -27.0), RATE, cap_seconds=21.93,
                                   ended_itself=False)
        assert long_but_ended.valid and not short_but_cut.valid

    def test_both_reasons_are_reported_when_both_apply(self):
        reason = check_clip(tone(21.93, -55.0), RATE, cap_seconds=21.93).reason
        assert "silent" in reason and "token cap" in reason

    def test_a_nonsense_sample_rate_is_refused(self):
        with pytest.raises(ValueError):
            check_clip(tone(1.0, -20.0), 0)


class TestSummary:
    def test_invalid_clips_are_counted_as_their_own_rate(self):
        checks = [check_clip(tone(3.0, -28.0), RATE)] * 38 + \
                 [check_clip(tone(21.93, -55.0), RATE, cap_seconds=21.93)] * 2
        summary = summarise(checks)
        assert (summary.clips, summary.valid, summary.invalid) == (40, 38, 2)
        assert summary.invalid_rate == pytest.approx(0.05)
        assert "2 silent" in summary.line() and "38/40" in summary.line()

    def test_an_empty_eval_does_not_divide_by_zero(self):
        assert summarise([]).invalid_rate == 0.0


# ---- the pilot's own audio -----------------------------------------------------------

PILOT = Path(r"E:\Geetha\runs\m2pre_pilot_eval\clips")
SILENT_BY_EAR = {"breath_3_with", "robustness_245", "robustness_255"}
# Audible (-30.5 dBFS) but 21.93 s for a five-word sentence: a runaway, not a mute. The
# 6 Nov analysis said "the clips that failed to terminate are exactly the silent ones";
# this clip is the counter-example, found by the gate on its first run.
RUNAWAY = {"breath_3_without"}


@pytest.mark.skipif(not PILOT.exists(), reason="pilot clips are not on this machine")
def test_the_gate_finds_the_clips_the_operator_found_by_ear_and_one_he_could_not():
    import soundfile as sf

    silent, capped, count = set(), set(), 0
    for path in sorted(PILOT.glob("*.wav")):
        samples, rate = sf.read(path, dtype="float32")
        count += 1
        check = check_clip(samples, rate, cap_seconds=21.93)
        if check.silent:
            silent.add(path.stem)
        if check.hit_cap:
            capped.add(path.stem)
    assert count == 53
    assert silent == SILENT_BY_EAR
    assert capped == SILENT_BY_EAR | RUNAWAY
