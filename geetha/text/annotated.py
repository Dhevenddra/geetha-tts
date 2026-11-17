"""Parsing DiarBench Tamil's inline dual-script annotation.

DiarBench writes every English loanword as the Tamil-script form it was actually spoken
as, followed by the source form in a bracket, with any Tamil case suffix *after* the
bracket::

    நேச்சுரல்(Natural) டோன்(Tone)ல நார்மல்(Normal)ஆ

That is a dual-script reference already — 12,813 pairs across 2,917 distinct native
forms, measured 26 Oct 2026 — so task B3 harvests it instead of transliterating
anything. This module converts it to the ``{native|latin}`` alternation convention that
``geetha.scoring.dual_script`` understands, and hands back the pairs so the same lexicon
can be applied to SPRING-INX, whose references carry bare Latin with no Tamil form.

**The corpus mixes its conventions, and the bracket shape does not tell you which is
which — only the word inside does.** All of the following are in the same file:

* nonverbal tags as ``<laughter>`` (13 times) *and* as ``[Laughter]`` (448 times);
* loanword annotation as ``ஓகே(okay)`` *and* as ``சோ[so]`` (~833 times);
* 363 annotations with a zero-width non-joiner wedged in before the bracket, which
  silently defeats a naive pattern and leaves ``(correct)`` sitting in the reference;
* three misspellings of "unintelligible" and two spellings of "background speech".

A bracketed token is a tag if its name is in ``NONVERBAL_TAGS``; otherwise it is a
loanword annotation. Getting this wrong is not cosmetic: ``[Laughter]`` left in a
reference becomes the literal word "laughter", i.e. a word the ASR is scored for failing
to say, 448 times over.

The parenthetical is not always Latin: 1,190 of the pairs hold a numeral (``ஒரு(1)``,
``ரெண்டு(2)``), and a handful hold ``10,000``, ``7:00`` or ``1/2``. All of them are
legitimate alternative renderings of what was said, so all become alternations.

Tags are pulled out of the reference rather than scored as words, and a segment left with
no words is marked not scorable — an unintelligible segment has no reference, and scoring
against an empty one would manufacture insertion errors out of nothing.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "AnnotatedSegment",
    "NONVERBAL_TAGS",
    "TAG_SPELLINGS",
    "canonical_tag",
    "harvest_lexicon",
    "parse_annotated",
]

TAMIL = r"஀-௿"

# Zero-width characters carry no sound but break every pattern they sit in. Measured in
# DiarBench Tamil: 363 ZWNJ, 1 ZWSP. Stripped before anything else is attempted.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
_NBSP = {0x00A0: " "}

# Nonverbal annotations, in either bracket shape. Anything bracketed whose name is NOT
# in here is a loanword annotation, not a tag.
NONVERBAL_TAGS = frozenset(
    {
        "unintelligible",
        "vocalization",
        "laughter",
        "stutters",
        "noise",
        "gasps",
        "coughing",
        "background_speech",
        "throat_clearing",
        "swallowing",
        "breathing",
    }
)

# The corpus's own spelling variants, folded to one canonical name so a judge is never
# calibrated against a typo.
TAG_SPELLINGS = {
    "unintelligble": "unintelligible",
    "unintellegible": "unintelligible",
    "unitelligible": "unintelligible",
    "background speech": "background_speech",
    "throat clearing": "throat_clearing",
    "vocalisation": "vocalization",
    "stutter": "stutters",
    "gasp": "gasps",
}

# native form, bracket, optional Tamil case suffix glued on after the bracket.
#
# Three tolerances, each for something the real corpus does:
#   * either bracket shape, and mismatched pairs -- "மீ (me]" is a transcription typo
#     for "மீ (me)", and left unparsed it puts the literal "(me]" into a reference;
#   * any run of whitespace, and sentence punctuation, between the word and the bracket
#     ("ம்ம்ம் (Mmm)", "டூ  (to)", "டீச்சரா?(teacher)", "ஸ்வீட். (sweet)"). Safe here
#     because a bracket is never ordinary punctuation in this corpus, only annotation;
#   * a case suffix that starts with a hyphen ("(teacher)-ரா", "(fifteen)-ல").
#
# Residual after these tolerances: 0.1% of segments, all genuine source irregularities.
_PAIR = re.compile(
    rf"([{TAMIL}][{TAMIL}\-]*)[\s.,?!]*[(\[]([^()\[\]]{{1,40}})[)\]](-?[{TAMIL}]*)"
)
_BRACKETED = re.compile(r"\[([^\[\]]{1,40})\]|<([^<>]{1,40})>")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class AnnotatedSegment:
    """One parsed transcript segment."""

    reference: str
    pairs: list[tuple[str, str]]
    tags: list[str]
    scorable: bool


def canonical_tag(raw: str) -> str:
    """Canonical name for a tag, folding the corpus's misspellings and spacing."""
    key = raw.strip().lower()
    return TAG_SPELLINGS.get(key, key.replace(" ", "_"))


def _is_tag(raw: str) -> bool:
    return canonical_tag(raw) in NONVERBAL_TAGS


def parse_annotated(text: str) -> AnnotatedSegment:
    """Convert one DiarBench transcript segment into a dual-script reference."""
    if not text:
        return AnnotatedSegment(reference="", pairs=[], tags=[], scorable=False)
    if any(ch in text for ch in "{|}"):
        raise ValueError(
            f"source text contains alternation syntax ({{, | or }}) and cannot be turned "
            f"into a dual-script reference safely: {text!r}"
        )

    text = text.translate(_INVISIBLE).translate(_NBSP)

    # 1. Nonverbal tags out first, in both bracket shapes. Loanword annotations in
    #    square brackets are left in place for step 2 to convert.
    tags: list[str] = []

    def take_tag(match: re.Match[str]) -> str:
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        if _is_tag(raw):
            tags.append(canonical_tag(raw))
            return " "
        return match.group(0)

    text = _BRACKETED.sub(take_tag, text)

    # 2. Loanword annotations become {native|source} alternations.
    pairs: list[tuple[str, str]] = []

    def take_pair(match: re.Match[str]) -> str:
        native, source, suffix = match.group(1), match.group(2).strip().lower(), match.group(3)
        pairs.append((native, source))
        return f"{{{native}|{source}}}{suffix}"

    text = _PAIR.sub(take_pair, text)

    # 3. A bracketed source form with no Tamil word before it still names a word that
    #    was spoken, so keep the word and drop only the brackets.
    text = _BRACKETED.sub(
        lambda m: (m.group(1) if m.group(1) is not None else m.group(2)).strip().lower(), text
    )

    reference = _WHITESPACE.sub(" ", text).strip()
    return AnnotatedSegment(
        reference=reference,
        pairs=pairs,
        tags=tags,
        scorable=bool(reference),
    )


def harvest_lexicon(segments: Iterable[AnnotatedSegment]) -> dict[str, list[str]]:
    """Map each source form to the native spellings seen for it, commonest first.

    Several Tamil spellings of one English word are normal and all of them are real
    (``ஓகே`` and ``ஒகே`` for "okay"), so none is discarded here — they become
    alternatives when the lexicon is applied to a reference carrying only the Latin
    form. The corpus does contain annotation errors (``ஓகே[so]``) and native forms with
    a case suffix baked in (``ஃபர்ஸ்ட்ல`` for "first"), so callers that apply this
    lexicon to another corpus should filter it by frequency first.
    """
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for segment in segments:
        for native, source in segment.pairs:
            counts[source][native] += 1
    return {
        source: [native for native, _ in counter.most_common()]
        for source, counter in counts.items()
    }
