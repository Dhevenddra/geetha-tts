"""Tests for geetha.eval.directability — building the Directability-Ta prompt set (B4).

Three kinds of item, each measuring something different:

* **carrier** — emotion-neutral text, rendered under every emotion label. The same text
  under different tags means any difference in delivery is attributable to the tag and
  not to the words, which is the only way to measure *directability* rather than the
  model's reading of loaded text.
* **emotional** — text whose content implies a feeling, rendered with **no** tag. Tests
  D-004: untagged text must still sound right.
* **probe** — a short phrase whose intent changes with delivery. Listeners pick the
  intent from a fixed list, so chance is 1/len(intents) and is stated per probe.
"""
import pytest

from geetha.eval.directability import (
    CARRIER_STYLES,
    SourceItem,
    build_items,
    estimate_seconds,
    find_leaks,
    load_source,
    validate_source,
)

SOURCE = load_source()


def test_the_source_parses_and_skips_comment_lines():
    assert all(not item.id.startswith("#") for item in SOURCE)
    assert len(SOURCE) == 140


def test_the_source_has_the_counts_the_runbook_asks_for():
    counts = {kind: sum(1 for i in SOURCE if i.kind == kind) for kind in ("carrier", "emotional", "probe")}
    assert counts == {"carrier": 40, "emotional": 60, "probe": 40}


def test_every_item_has_text_and_a_gloss():
    for item in SOURCE:
        assert item.text.strip(), item.id
        assert item.gloss.strip(), item.id


def test_ids_are_unique():
    ids = [item.id for item in SOURCE]
    assert len(ids) == len(set(ids))


def test_carriers_and_emotional_items_declare_no_intents():
    for item in SOURCE:
        if item.kind in ("carrier", "emotional"):
            assert item.intents == (), item.id


def test_every_probe_offers_at_least_two_intents():
    for item in SOURCE:
        if item.kind == "probe":
            assert len(item.intents) >= 2, item.id


def test_carrier_styles_are_five_trained_uppercase_labels():
    # 40 carriers x 5 styles = the 200 tagged prompts runbook B4 asks for. DISGUST is
    # left out of the carriers and covered by the untagged emotional items instead.
    assert len(CARRIER_STYLES) == 5
    assert CARRIER_STYLES == ("HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE")


def test_building_produces_the_runbooks_three_hundred_prompts():
    items = build_items(SOURCE)
    kinds = {k: sum(1 for i in items if i["kind"] == k) for k in ("tagged", "untagged_emotional", "probe")}
    assert kinds == {"tagged": 200, "untagged_emotional": 60, "probe": 40}
    assert len(items) == 300


def test_a_tagged_prompt_carries_its_style_and_its_carrier_id():
    items = build_items(SOURCE)
    tagged = [i for i in items if i["kind"] == "tagged"]
    first = tagged[0]
    assert first["style"] in CARRIER_STYLES
    assert first["carrier_id"] == "c01"
    assert first["id"] == "c01/HAPPY"


def test_every_carrier_appears_under_every_style():
    items = build_items(SOURCE)
    pairs = {(i["carrier_id"], i["style"]) for i in items if i["kind"] == "tagged"}
    assert len(pairs) == 200


def test_untagged_emotional_prompts_carry_no_style():
    items = build_items(SOURCE)
    for item in items:
        if item["kind"] == "untagged_emotional":
            assert item["style"] is None, item["id"]


def test_probes_carry_their_intents_and_the_chance_rate():
    items = build_items(SOURCE)
    probe = next(i for i in items if i["kind"] == "probe")
    assert len(probe["intents"]) >= 2
    assert probe["chance"] == pytest.approx(1 / len(probe["intents"]))


def test_duration_is_estimated_from_the_voices_measured_rate():
    # voices.md: Anitha 13.2 chars/sec, Arun 14.5. Script length is planned against
    # the rate, so a prompt that would overrun the 20 s clip budget must be visible.
    assert estimate_seconds("a" * 132, chars_per_second=13.2) == pytest.approx(10.0)


def test_no_authored_prompt_is_long_enough_to_overrun_the_clip_budget():
    # 20 s is the clip-split threshold (82 tokens/s, 2,048-token sequence).
    for item in SOURCE:
        assert estimate_seconds(item.text, chars_per_second=13.2) < 20.0, item.id


def test_find_leaks_reports_a_prompt_that_appears_in_training_text():
    index = {"வணக்கம் நண்பா"}
    items = [
        SourceItem(kind="carrier", id="x1", text="வணக்கம், நண்பா!", gloss="g", intents=()),
        SourceItem(kind="carrier", id="x2", text="வேறு வாக்கியம்", gloss="g", intents=()),
    ]
    # The leak hash sees through punctuation, so x1 collides and x2 does not.
    assert find_leaks(items, index) == [("x1", "வணக்கம், நண்பா!")]


def test_find_leaks_returns_nothing_for_a_clean_set():
    index = {"முற்றிலும் வேறு"}
    items = [SourceItem(kind="carrier", id="x1", text="வணக்கம்", gloss="g", intents=())]
    assert find_leaks(items, index) == []


def test_validate_source_accepts_the_authored_set():
    validate_source(SOURCE)  # raises if anything is wrong


def test_validate_source_rejects_a_probe_with_one_intent():
    bad = [SourceItem(kind="probe", id="p99", text="சரி", gloss="ok", intents=("agreement",))]
    with pytest.raises(ValueError, match="p99"):
        validate_source(bad)


def test_validate_source_rejects_a_duplicate_id():
    dup = [
        SourceItem(kind="carrier", id="c01", text="a", gloss="g", intents=()),
        SourceItem(kind="carrier", id="c01", text="b", gloss="g", intents=()),
    ]
    with pytest.raises(ValueError, match="c01"):
        validate_source(dup)


def test_validate_source_rejects_duplicate_text():
    # Two carriers with the same text would silently halve a comparison.
    dup = [
        SourceItem(kind="carrier", id="c01", text="ஒரே வாக்கியம்", gloss="g", intents=()),
        SourceItem(kind="carrier", id="c02", text="ஒரே வாக்கியம்.", gloss="g", intents=()),
    ]
    with pytest.raises(ValueError, match="same normalised text"):
        validate_source(dup)
