import pytest

from geetha.audio.align import WordSpan, boundary_for_span, place_tags, romanise_words

WORDS = [WordSpan("a", 0.5, 1.0, 0.9), WordSpan("b", 1.1, 1.6, 0.9), WordSpan("c", 2.4, 3.0, 0.9)]


def test_span_before_the_first_word_is_a_lead():
    placement = boundary_for_span(WORDS, 0.0, 0.34, 3.5)
    assert placement.before_word == 0 and placement.covered == pytest.approx(1.0)


def test_span_in_a_mid_gap_names_the_word_after_it():
    placement = boundary_for_span(WORDS, 1.8, 2.2, 3.5)
    assert placement.before_word == 2 and placement.covered == pytest.approx(1.0)


def test_span_inside_a_word_is_poorly_covered():
    assert boundary_for_span(WORDS, 0.6, 0.9, 3.5).covered == 0.0


def test_span_after_the_last_word_is_trailing():
    assert boundary_for_span(WORDS, 3.1, 3.4, 3.5).before_word == len(WORDS)


def test_empty_span_is_refused():
    with pytest.raises(ValueError):
        boundary_for_span(WORDS, 1.0, 1.0, 3.5)


def test_place_tags_writes_each_position_once():
    assert place_tags(["a", "b", "c"], [0, 2, 2], "<T>") == "<T> a b <T> c"
    assert place_tags(["a", "b"], [2], "<T>") == "a b <T>"
    with pytest.raises(ValueError):
        place_tags(["a"], [3], "<T>")


def test_romanise_keeps_letters_only():
    assert romanise_words(["X1", "y z"], lambda w: w) == ["x", "yz"]
