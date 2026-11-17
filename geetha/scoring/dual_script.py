"""Dual-script scoring for code-mixed Tamil references (runbook sections 1 and 6).

A reference writes its script-ambiguous spans as alternations::

    "{ஓகே|okay} {மேம்|ma'am} {டூ|two} நிமிஷம்"

Every reading inside the braces is something a Tamil transcriber would legitimately
write, so the scorer takes whichever reading the hypothesis matches and reports which.
A correct transcription is never penalised for a script choice.

**The reference is scored as a lattice, not as a list of strings.** Real DiarBench
segments carry fourteen or more code-mixed words, whose cartesian product is 16,384
references, so scoring aligns the hypothesis against the lattice with dynamic
programming instead (``geetha.scoring.lattice``): exact, and linear in the number of
sites. It is also what makes a *mixed* hypothesis score correctly — a transcriber who
writes some loanwords in Tamil and some in Latin within one sentence is right either
way, and the reference must accept the mixture rather than force one script across the
whole utterance.

``expand_variants`` is kept for small references, for inspection, and for the property
test that checks the lattice result against brute force.

This is a scoring allowance, not a licence to be vague: only spans a Tamil speaker would
genuinely write either way belong in braces. Every reading actually chosen is reported,
so the released raw outputs show what each utterance was scored against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from geetha.scoring.lattice import build_lattice, lattice_edit_counts
from geetha.scoring.metrics import ErrorCounts, cer, grapheme_cer, graphemes, wer
from geetha.text.normalise import norm_text

__all__ = [
    "DualScriptResult",
    "MAX_VARIANTS",
    "expand_variants",
    "parse_alternations",
    "score_dual",
]

MAX_VARIANTS = 256

_ALTERNATION = re.compile(r"\{([^{}]*)\}")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DualScriptResult:
    """The counts for the best reading found, and the reference text it used."""

    counts: ErrorCounts
    variant: str


def parse_alternations(reference: str) -> list[str | list[str]]:
    """Split a reference into literal strings and lists of alternative readings."""
    parts: list[str | list[str]] = []
    position = 0
    for match in _ALTERNATION.finditer(reference):
        if match.start() > position:
            parts.append(reference[position : match.start()])
        parts.append(match.group(1).split("|"))
        position = match.end()
    if position < len(reference):
        parts.append(reference[position:])
    return parts or [""]


def _render(parts: list[str | list[str]], choices: list[int]) -> str:
    out: list[str] = []
    site = 0
    for part in parts:
        if isinstance(part, list):
            out.append(part[choices[site]])
            site += 1
        else:
            out.append(part)
    # An empty reading leaves a double space behind; tidy it so the rendered reference
    # reads as the sentence it represents.
    return _WHITESPACE.sub(" ", "".join(out)).strip()


def expand_variants(reference: str) -> list[str]:
    """Every combination of readings, for small references and for inspection.

    Raises ``ValueError`` past ``MAX_VARIANTS``. ``score_dual`` does not use this --
    it scores the lattice directly -- so a crowded reference still scores fine.
    """
    parts = parse_alternations(reference)
    sites = [p for p in parts if isinstance(p, list)]

    total = 1
    for options in sites:
        total *= len(options)
        if total > MAX_VARIANTS:
            raise ValueError(
                f"reference expands to more than {MAX_VARIANTS} variants: {reference!r}"
            )

    combinations: list[list[int]] = [[]]
    for options in sites:
        combinations = [
            combo + [index] for combo in combinations for index in range(len(options))
        ]
    return [_render(parts, combo) for combo in combinations]


def _chunks(reference: str) -> list[str]:
    """Split a reference on whitespace, never inside an alternation."""
    chunks: list[str] = []
    current: list[str] = []
    depth = 0
    for char in reference:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        if char.isspace() and depth == 0:
            if current:
                chunks.append("".join(current))
                current = []
        else:
            current.append(char)
    if current:
        chunks.append("".join(current))
    return chunks


def _chunk_readings(chunk: str) -> list[str]:
    """Every reading of one chunk.

    Expansion is within the chunk only -- a single word rarely holds more than one
    alternation -- so this stays tiny however many sites the whole reference has.
    Order is preserved and duplicates are dropped, which keeps tie-breaking stable.
    """
    parts = parse_alternations(chunk)
    sites = [p for p in parts if isinstance(p, list)]
    if not sites:
        return [chunk]
    combinations: list[list[int]] = [[]]
    for options in sites:
        combinations = [c + [i] for c in combinations for i in range(len(options))]
    readings: list[str] = []
    for combo in combinations:
        rendered = _render(parts, combo)
        if rendered not in readings:
            readings.append(rendered)
    return readings


def _slots(reference: str, tokenise, separator: str | None) -> list[list[list[str]]]:
    """One slot per chunk, each slot holding the token sequence of every reading.

    ``separator`` is the token inserted between chunks for units that count spaces
    (CER does; WER does not, because its tokens *are* the words).
    """
    slots: list[list[list[str]]] = []
    for index, chunk in enumerate(_chunks(reference)):
        if index and separator is not None:
            slots.append([[separator]])
        readings = [tokenise(r) for r in _chunk_readings(chunk)]
        deduped: list[list[str]] = []
        for tokens in readings:
            if tokens not in deduped:
                deduped.append(tokens)
        slots.append(deduped or [[]])
    return slots


def _word_tokens(text: str) -> list[str]:
    return norm_text(text).split()


def _char_tokens(text: str) -> list[str]:
    return list(norm_text(text))


def _grapheme_tokens(text: str) -> list[str]:
    return graphemes(norm_text(text))


# Each supported metric and the unit its lattice is built from. Keyed by the metric
# function itself so callers keep passing ``metric=cer`` as before.
_TOKENISERS: dict[Callable[[str, str], ErrorCounts], Callable[[str], list[str]]] = {
    wer: _word_tokens,
    cer: _char_tokens,
    grapheme_cer: _grapheme_tokens,
}


def score_dual(
    hypothesis: str,
    reference: str,
    metric: Callable[[str, str], ErrorCounts] = wer,
) -> DualScriptResult:
    """Score ``hypothesis`` against the cheapest reading of the reference lattice.

    Exact: the alignment and the choice of reading at every site are solved together,
    so no site is resolved on a guess about its neighbours.
    """
    try:
        tokenise = _TOKENISERS[metric]
    except KeyError:
        raise ValueError(
            f"score_dual supports wer, cer and grapheme_cer; got {getattr(metric, '__name__', metric)!r}"
        ) from None

    separator = None if metric is wer else " "
    lattice = build_lattice(_slots(reference, tokenise, separator))
    counts, chosen = lattice_edit_counts(tokenise(hypothesis), lattice)

    joiner = " " if metric is wer else ""
    return DualScriptResult(counts, joiner.join(chosen))
