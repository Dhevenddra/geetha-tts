r"""Regression lock for norm_text() over 200 real Tamil sentences.

These are not fresh assertions about what normalisation *should* do -- every rule
was driven by a failing test in test_normalise.py. This file freezes the result of
those rules on real text drawn from the five corpora we train and test on
(data/norm_regression_200.tsv, sampled with seed 1234 by
scripts/make_norm_review.py).

Why it matters: norm_text defines the leak hash of runbook rule 3. If it changes,
every manifest hash and the frozen eval vault change with it. This test makes that
impossible to do by accident -- a change here must be a deliberate edit to the
fixture, with a note saying why.

The operator reviews the same 200 sentences in E:\Geetha\work\norm_review_200.tsv
(task B2's done-when).
"""
import csv
from pathlib import Path

import pytest

from geetha.text.normalise import norm_text

FIXTURE = Path(__file__).parent.parent / "data" / "norm_regression_200.tsv"


def load_cases() -> list[tuple[str, str, str]]:
    with FIXTURE.open(encoding="utf-8", newline="") as handle:
        return [
            (row["source"], row["original"], row["normalised"])
            for row in csv.DictReader(handle, delimiter="\t")
        ]


CASES = load_cases()


def test_the_fixture_holds_two_hundred_sentences():
    assert len(CASES) == 200


def test_every_corpus_is_represented():
    assert {source for source, _, _ in CASES} == {
        "rasa_test",
        "indicvoices_unsanitized",
        "indicvoices_r_test",
        "fleurs_test",
        "spring_inx_r1_test",
    }


@pytest.mark.parametrize(
    "source,original,expected",
    CASES,
    ids=[f"{i:03d}-{source}" for i, (source, _, _) in enumerate(CASES)],
)
def test_normalisation_is_unchanged(source: str, original: str, expected: str):
    assert norm_text(original) == expected


def test_normalisation_is_idempotent_on_every_sentence():
    # Normalising twice must equal normalising once, or a manifest built in two
    # passes disagrees with one built in a single pass.
    for _, original, _ in CASES:
        once = norm_text(original)
        assert norm_text(once) == once
