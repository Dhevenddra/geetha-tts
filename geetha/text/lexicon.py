"""Applying a harvested Tamil/Latin lexicon to references that carry only Latin.

SPRING-INX Tamil test references write English in Latin script with no Tamil form
("okay ma'am", "so correct") — 54-61% of their utterances are code-mixed this way.
DiarBench's inline annotation gives us 3,206 Latin-to-Tamil mappings for nothing (see
``geetha.text.annotated``), so those bare Latin tokens become ``{latin|native}``
alternations and an ASR that renders an English word in Tamil script is no longer scored
wrong for a script choice that a Tamil transcriber would also have made.

Two guards, because a too-lenient reference is a worse failure than a too-strict one:

* ``filter_lexicon`` drops spellings seen only once or twice. The corpus contains real
  annotation errors — ``ஓகே`` tagged as "so" — and importing those into a reference
  would quietly hand us free credit on a published benchmark.
* ``MAX_ALTERNATIVES`` caps how many Tamil spellings one token may offer. Not for
  tractability -- ``score_dual`` scores the reference as a lattice and handles any
  number of sites -- but because the tail of rare spellings is where the annotation
  errors live, and each extra reading can only ever make a reference more lenient.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

__all__ = ["MAX_ALTERNATIVES", "apply_lexicon", "filter_lexicon"]

# Tamil spellings offered per token. Three covers the genuine variation seen in the
# corpus (ஓகே / ஒகே / ஓக்கே for "okay"); past that it is annotation noise.
MAX_ALTERNATIVES = 3

# A Latin word, with any attached punctuation left outside the match.
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def filter_lexicon(
    lexicon: Mapping[str, Sequence[str]],
    counts: Mapping[tuple[str, str], int],
    *,
    min_count: int = 2,
) -> dict[str, list[str]]:
    """Keep only native spellings seen at least ``min_count`` times.

    A source form left with no surviving spelling is dropped entirely.
    """
    filtered: dict[str, list[str]] = {}
    for source, natives in lexicon.items():
        kept = [n for n in natives if counts.get((source, n), 0) >= min_count]
        if kept:
            filtered[source] = kept
    return filtered


def apply_lexicon(
    reference: str,
    lexicon: Mapping[str, Sequence[str]],
    *,
    max_alternatives: int = MAX_ALTERNATIVES,
) -> str:
    """Turn every known Latin word in ``reference`` into a dual-script alternation."""
    if any(ch in reference for ch in "{|}"):
        raise ValueError(
            f"reference already contains alternation syntax and must not be rewritten: "
            f"{reference!r}"
        )
    if not lexicon:
        return reference

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        natives = lexicon.get(word.lower())
        if not natives:
            return word
        options = "|".join([word, *natives[:max_alternatives]])
        return f"{{{options}}}"

    return _LATIN_WORD.sub(replace, reference)
