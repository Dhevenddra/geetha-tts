"""Tests for geetha.eval.breath_calibration — choosing the clips that set the threshold.

The detector's threshold is what turns audio into the headline number, and this module
decides which real clips it is calibrated on. Two mistakes here would quietly bias
everything downstream: putting a noisy clip in (the detector learns to fire on noise)
or putting a clip with a filler but no breath into the *negative* set (a filler sounds
enough like a breath that it would be scored as a false positive the detector never
really made).
"""
import pytest

from geetha.eval.breath_calibration import (
    AMBIGUOUS,
    BREATH_MARKS,
    MAX_SECONDS,
    MIN_SECONDS,
    MIN_SNR_DB,
    NEGATIVE,
    POSITIVE,
    count_marks,
    label_for,
    rejection_reason,
)


# --- reading the marks -----------------------------------------------------------


def test_breathing_marks_are_counted():
    text = "ஒன்று [breathing] இரண்டு [breathing] மூன்று"
    assert count_marks(text)["<|nv_breath|>"] == 2


def test_corpus_names_are_mapped_to_the_models_token_names():
    # The corpus says [throat_clearing]; the model's token is <|nv_throat|>. Using two
    # vocabularies for one event is how a count silently splits in half.
    assert count_marks("அ [throat_clearing] ஆ")["<|nv_throat|>"] == 1
    assert count_marks("அ [laughter] ஆ")["<|nv_laugh|>"] == 1


def test_filler_marks_are_counted_separately_from_non_verbal_events():
    # [uhh] has no token (contract.py: the closed set holds no filler token), so it is
    # speech, not an acoustic event. It must not land in the breath count.
    marks = count_marks("அ [uhh] ஆ [umm] இ")
    assert marks.get("<|nv_breath|>", 0) == 0
    assert marks["filler"] == 2


def test_noise_markers_are_recognised_as_neither():
    marks = count_marks("[Persistent-noise-start] அ ஆ [Persistent-noise-end]")
    assert marks.get("<|nv_breath|>", 0) == 0
    assert marks["noise"] >= 1


def test_text_with_no_marks_has_no_counts():
    assert count_marks("ஒன்று இரண்டு மூன்று") == {}


def test_empty_text_is_handled():
    assert count_marks("") == {}
    assert count_marks(None) == {}


def test_the_breath_marks_set_is_stated_so_a_report_can_quote_it():
    # Which events count as "a breath" for the headline is a choice, not a fact.
    assert "<|nv_breath|>" in BREATH_MARKS


# --- labelling -------------------------------------------------------------------


def test_a_clip_with_a_breathing_mark_is_positive():
    assert label_for(count_marks("அ [breathing] ஆ")) == POSITIVE


def test_a_clip_with_no_marks_at_all_is_negative():
    assert label_for(count_marks("அ ஆ இ")) == NEGATIVE


def test_a_clip_with_a_filler_but_no_breath_is_ambiguous_not_negative():
    # The important one. A hesitation is breathy; scoring it as a clean negative would
    # charge the detector with a false positive it did not really make, which pushes
    # the calibrated threshold too high and then under-counts our own breaths.
    assert label_for(count_marks("அ [uhh] ஆ")) == AMBIGUOUS


def test_a_clip_with_another_non_verbal_event_but_no_breath_is_ambiguous():
    for mark in ("[swallowing]", "[throat_clearing]", "[laughter]", "[cough]"):
        assert label_for(count_marks(f"அ {mark} ஆ")) == AMBIGUOUS, mark


def test_a_clip_with_a_breath_and_a_filler_is_still_positive():
    # Positives only need a breath to be present; other events do not disqualify them.
    assert label_for(count_marks("அ [uhh] ஆ [breathing] இ")) == POSITIVE


def test_a_clip_carrying_only_a_noise_marker_is_ambiguous():
    assert label_for(count_marks("[Persistent-noise-start] அ ஆ")) == AMBIGUOUS


# --- rejecting unusable clips ----------------------------------------------------


def _row(seconds=6.0, snr=40.0, text="அ ஆ"):
    return {"duration": seconds, "snr": snr, "unsanitized_verbatim": text}


def test_a_clean_clip_is_accepted():
    assert rejection_reason(_row()) is None


def test_a_clip_below_the_snr_floor_is_rejected():
    reason = rejection_reason(_row(snr=MIN_SNR_DB - 1))
    assert reason is not None and "snr" in reason.lower()


def test_a_clip_carrying_persistent_noise_is_rejected_whatever_its_snr():
    # The annotators flagged the noise; the snr figure did not notice it.
    reason = rejection_reason(_row(snr=60.0, text="[Persistent-noise-start] அ ஆ"))
    assert reason is not None and "noise" in reason.lower()


def test_a_clip_too_short_to_hold_an_event_is_rejected():
    reason = rejection_reason(_row(seconds=MIN_SECONDS - 0.1))
    assert reason is not None and "short" in reason.lower()


def test_a_clip_too_long_is_rejected():
    # Long clips dominate a per-clip rate and make the duration denominator lumpy.
    reason = rejection_reason(_row(seconds=MAX_SECONDS + 0.1))
    assert reason is not None and "long" in reason.lower()


def test_a_missing_snr_is_rejected_rather_than_assumed_good():
    reason = rejection_reason({"duration": 6.0, "snr": None, "unsanitized_verbatim": "அ"})
    assert reason is not None


def test_the_thresholds_are_named_constants():
    assert MIN_SNR_DB > 0
    assert 0 < MIN_SECONDS < MAX_SECONDS


def test_rejection_reasons_are_short_enough_to_tabulate():
    # They are counted into the report, so they must be a small closed set, not prose.
    reasons = {
        rejection_reason(_row(snr=1.0)),
        rejection_reason(_row(seconds=0.5)),
        rejection_reason(_row(seconds=999.0)),
        rejection_reason(_row(text="[noise] அ")),
    }
    for reason in reasons:
        assert reason is not None
        assert len(reason) < 40, reason


@pytest.mark.parametrize("mark", ["[breathing]", "[BREATHING]", "[Breathing]"])
def test_mark_matching_is_case_insensitive(mark):
    # The corpus is hand-annotated; casing varies.
    assert count_marks(f"அ {mark} ஆ")["<|nv_breath|>"] == 1


# --- rejecting from precounted marks ---------------------------------------------
#
# The builder indexes IndicVoices once into mark *counts* rather than keeping every
# annotation string in memory, so the noise check has to work from those counts too.
# Reconstructing a fake annotation string to feed the text check would be a second,
# divergent code path for the same decision.


def test_noise_is_detected_from_precounted_marks():
    reason = rejection_reason({"duration": 6.0, "snr": 40.0,
                               "marks": {"noise": 1}})
    assert reason is not None and "noise" in reason.lower()


def test_clean_precounted_marks_are_accepted():
    assert rejection_reason({"duration": 6.0, "snr": 40.0,
                             "marks": {"<|nv_breath|>": 2}}) is None


def test_precounted_marks_and_text_agree_on_a_clean_clip():
    from_text = rejection_reason({"duration": 6.0, "snr": 40.0,
                                  "unsanitized_verbatim": "அ [breathing] ஆ"})
    from_marks = rejection_reason({"duration": 6.0, "snr": 40.0,
                                   "marks": count_marks("அ [breathing] ஆ")})
    assert from_text == from_marks is None


def test_precounted_marks_and_text_agree_on_a_noisy_clip():
    text = "[Persistent-noise-start] அ ஆ"
    assert rejection_reason({"duration": 6.0, "snr": 40.0,
                             "unsanitized_verbatim": text}) is not None
    assert rejection_reason({"duration": 6.0, "snr": 40.0,
                             "marks": count_marks(text)}) is not None


def test_snr_and_duration_still_apply_with_precounted_marks():
    assert rejection_reason({"duration": 0.5, "snr": 40.0, "marks": {}}) is not None
    assert rejection_reason({"duration": 6.0, "snr": 1.0, "marks": {}}) is not None
