"""Tests for geetha.eval.capability — the Naturalness-Ta CAPABILITY component.

This is the component D-021 made the release headline, so the tests are about keeping
the claim exactly as strong as the evidence and no stronger.
"""
import pytest

from geetha.eval.capability import (
    CONDITIONS,
    CapabilityItem,
    event_rate_per_minute,
    false_positive_rate,
    marks_in,
    render_text,
    spoken_text,
    validate_items,
)


def _item(**overrides) -> CapabilityItem:
    base = dict(
        kind="event",
        id="ev01",
        text="நான் <|nv_breath|> யோசித்துப் பார்த்தேன்.",
        gloss="I thought about it.",
        expected_events=1,
    )
    base.update(overrides)
    return CapabilityItem(**base)


# --- what counts as a mark ------------------------------------------------------


def test_marks_are_read_out_of_the_text_in_order():
    item = _item(text="<|nv_breath|> சரி <|nv_sigh|> போகலாம்.", expected_events=2)
    assert marks_in(item.text) == ["<|nv_breath|>", "<|nv_sigh|>"]


def test_an_invented_token_is_rejected_rather_than_rendered():
    # D-017: no new tokens. A token the tokenizer does not hold is silently read as
    # literal text, so the clip would contain someone saying "nv underscore uh".
    with pytest.raises(ValueError, match="not in the model's token contract"):
        validate_items([_item(text="நான் <|nv_uh|> வந்தேன்.", expected_events=1)])


def test_expected_events_must_match_the_marks_actually_present():
    with pytest.raises(ValueError, match="expected_events"):
        validate_items([_item(expected_events=3)])


# --- the clean control ----------------------------------------------------------


def test_a_clean_item_carrying_a_mark_is_rejected():
    # The clean set is what turns "Geetha breathes" into a measurement rather than a
    # boast: it is where a breath is a *defect*. A mark in here would destroy that.
    with pytest.raises(ValueError, match="clean"):
        validate_items([_item(kind="clean", id="cl01", expected_events=0,
                              text="இன்று <|nv_breath|> வானிலை நன்றாக உள்ளது.")])


def test_a_clean_item_expects_zero_events():
    with pytest.raises(ValueError, match="clean"):
        validate_items([_item(kind="clean", id="cl01", text="வானிலை நன்றாக உள்ளது.",
                              expected_events=1)])


# --- filler items are lexical, not tokens ---------------------------------------


def test_a_filler_item_must_not_use_a_nonverbal_token():
    # contract.py: the closed token set has no filler token, so a hesitation is Tamil
    # text the model speaks. Writing a filler as a token would claim a capability gap
    # where there is none -- the base model can say "அதாவது" perfectly well.
    with pytest.raises(ValueError, match="filler"):
        validate_items([_item(kind="filler", id="fl01", expected_events=1,
                              text="அதாவது <|nv_breath|> எனக்கு தெரியல.")])


def test_a_filler_item_expects_no_nonverbal_events():
    item = _item(kind="filler", id="fl01", text="அதாவது... எனக்கு சரியா தெரியல.",
                 expected_events=0)
    validate_items([item])  # does not raise


# --- the two rendering conditions -----------------------------------------------


def test_the_marked_condition_keeps_the_tokens():
    item = _item()
    assert render_text(item, "marked") == item.text


def test_the_stripped_condition_removes_the_tokens_and_their_spacing():
    item = _item()
    assert render_text(item, "stripped") == "நான் யோசித்துப் பார்த்தேன்."


def test_every_baseline_system_can_only_be_given_the_stripped_condition():
    # A commercial API has no idea what <|nv_breath|> is; sending it would measure
    # their tokeniser's politeness, not their speech.
    assert CONDITIONS["stripped"].for_baselines is True
    assert CONDITIONS["marked"].for_baselines is False


def test_spoken_text_is_what_the_duration_budget_is_measured_against():
    # Marks cost characters but no speaking time; budgeting on the raw string would
    # under-fill every prompt.
    item = _item()
    assert spoken_text(item.text) == "நான் யோசித்துப் பார்த்தேன்."


# --- prompts must render in a single pass ---------------------------------------


def test_a_prompt_too_long_for_one_pass_is_rejected():
    # A forced mid-sentence split inserts a pause the writer did not write. For CER
    # that is harmless; for a naturalness prompt set it is self-inflicted damage, so
    # these prompts must fit one pass.
    long_text = "இது ஒரு மிக நீண்ட வாக்கியம் ஆகும். " * 12
    with pytest.raises(ValueError, match="single pass"):
        validate_items([_item(text=long_text, expected_events=0, kind="clean", id="cl02")])


# --- duplicates and glosses -----------------------------------------------------


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_items([_item(), _item()])


def test_duplicate_spoken_text_is_rejected():
    # Two items differing only in where a breath sits would double-count one sentence
    # in the distributional half.
    with pytest.raises(ValueError, match="duplicate"):
        validate_items([
            _item(id="ev01", text="நான் <|nv_breath|> வந்தேன்.", expected_events=1),
            _item(id="ev02", text="நான் வந்தேன் <|nv_sigh|>.", expected_events=1),
        ])


def test_an_item_without_a_gloss_is_rejected():
    with pytest.raises(ValueError, match="gloss"):
        validate_items([_item(gloss="   ")])


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        validate_items([_item(kind="breathy")])


# --- the measurements themselves ------------------------------------------------


def test_event_rate_is_events_per_minute_of_audio():
    assert event_rate_per_minute(events=6, seconds=120.0) == 3.0


def test_event_rate_of_a_system_that_emits_nothing_is_zero_not_undefined():
    # This is the number C2 predicts for the base model, so it must not divide by zero
    # or come back as None and get dropped from a table.
    assert event_rate_per_minute(events=0, seconds=90.0) == 0.0


def test_event_rate_rejects_zero_duration():
    with pytest.raises(ValueError):
        event_rate_per_minute(events=1, seconds=0.0)


def test_false_positive_rate_counts_clean_prompts_that_got_an_event():
    assert false_positive_rate(clean_with_events=3, clean_total=24) == pytest.approx(0.125)


def test_false_positive_rate_rejects_an_empty_clean_set():
    # Without a clean set the capability number is unfalsifiable: a model that breathes
    # constantly would score best.
    with pytest.raises(ValueError, match="clean"):
        false_positive_rate(clean_with_events=0, clean_total=0)


# --- the authored source itself -------------------------------------------------


def test_the_authored_source_loads_and_validates():
    # The real file, not a fixture: a prompt set that fails its own validator is a
    # measurement that cannot be run.
    from geetha.eval.capability import load_source

    items = load_source()
    validate_items(items)
    assert items


def test_the_authored_source_has_the_counts_the_report_will_quote():
    from collections import Counter

    from geetha.eval.capability import EXPECTED_COUNTS, load_source

    counts = Counter(item.kind for item in load_source())
    assert dict(counts) == EXPECTED_COUNTS


def test_every_event_item_carries_at_least_one_mark():
    from geetha.eval.capability import load_source

    for item in load_source():
        if item.kind == "event":
            assert item.expected_events >= 1, item.id


def test_no_clean_item_invites_a_hesitation():
    # A clean item containing an ellipsis is inviting exactly the behaviour it is
    # supposed to prove absent.
    from geetha.eval.capability import load_source

    for item in load_source():
        if item.kind == "clean":
            assert "..." not in item.text, item.id
            assert "…" not in item.text, item.id
