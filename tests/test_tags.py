"""Tests for geetha.text.tags — the inline direction tag inventory.

Runbook section 8 requires tags stripped before CER is computed. Stripping must be
explicit and loud: a tag we do not know about should stop the run, not vanish.
"""
import pytest

from geetha.text.contract import EMOTION_STYLES
from geetha.text.tags import EMOTION_TAGS, NONVERBAL_TAGS, strip_tags


def test_emotion_tag_is_removed_and_text_survives():
    assert strip_tags("<happy>வணக்கம்") == "வணக்கம்"


def test_nonverbal_tags_are_removed():
    assert strip_tags("வணக்கம் [breath] நண்பா [uh] வா") == "வணக்கம் நண்பா வா"


def test_emphasis_markers_go_but_the_emphasised_words_stay():
    assert strip_tags("இது [emphasis]மிகவும்[/emphasis] நல்லது") == "இது மிகவும் நல்லது"


def test_an_unknown_bracket_tag_raises_rather_than_being_dropped_silently():
    # A stray transcript artefact like [noise] must not quietly become a deletion
    # error in someone's CER number.
    with pytest.raises(ValueError, match=r"\[noise\]"):
        strip_tags("வணக்கம் [noise] நண்பா")


def test_unknown_tags_can_be_left_in_place_when_not_strict():
    assert strip_tags("வணக்கம் [noise]", strict=False) == "வணக்கம் [noise]"


def test_the_emotion_inventory_is_the_models_trained_labels_not_the_runbooks():
    # The runbook listed <happy> <sad> <angry> <fear> <surprise> <disgust> <neutral>.
    # voices.md gives the labels Indic-Speak was actually trained on: "anger" not
    # "angry", and no "neutral" at all. Conditioning on a label the model never saw
    # is silent misconditioning, so this inventory follows the model.
    assert EMOTION_TAGS == frozenset(EMOTION_STYLES)
    assert "angry" not in EMOTION_TAGS
    assert "neutral" not in EMOTION_TAGS


def test_the_runbooks_emotion_spellings_are_refused_loudly():
    with pytest.raises(ValueError, match="angry"):
        strip_tags("<angry>வணக்கம்")


def test_nonverbal_tags_cover_what_the_corpora_actually_mark():
    assert {"breath", "uh", "um", "tsk"} <= NONVERBAL_TAGS


def test_stripping_leaves_no_double_spaces():
    assert strip_tags("ஒரு [breath] இரண்டு") == "ஒரு இரண்டு"
