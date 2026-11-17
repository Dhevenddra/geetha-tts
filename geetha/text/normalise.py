"""Tamil text normalisation for Geetha v1.

``norm_text`` is the single canonical form used for two things, and they must stay
the same function or the leak rule (runbook rule 3) and the scorers disagree:

1. the leak hash that decides whether a sentence is in a held-out split;
2. the reference/hypothesis form fed to WER/CER.

Inline direction tags are *not* removed here. Strip them explicitly with
``strip_tags`` first (runbook section 8: "tags stripped before CER") so that a
tag leaking into a scored string is a visible bug rather than a silent one.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["norm_text"]

# Tamil digits U+0BE6..U+0BEF -> ASCII, so "௧௨" and "12" hash alike.
_TAMIL_DIGITS = {0x0BE6 + i: ord("0") + i for i in range(10)}

# Apostrophes are deleted rather than spaced: "don't" must stay one token.
_APOSTROPHES = dict.fromkeys(map(ord, "'‘’‛´ʼ`"))

# Invisible characters that carry no phonetic content but break hash equality.
# ZWJ/ZWNJ appear in Tamil web text; the BOM arrives with UTF-8-sig files.
_INVISIBLE = dict.fromkeys(
    map(
        ord,
        "​"  # zero width space
        "‌"  # zero width non-joiner
        "‍"  # zero width joiner
        "⁠"  # word joiner
        "﻿"  # byte order mark
        "­",  # soft hyphen
    )
)

_WHITESPACE = re.compile(r"\s+")

# A separator sitting between two digits is part of the number, not punctuation.
# Grouping commas are deleted so "1,00,000" and "100000" agree; the decimal point
# and the time colon are kept so "25.5" and "3:30" stay single tokens.
_GROUPING_COMMA = re.compile(r"(?<=\d),(?=\d)")
_INTRA_NUMBER = re.compile(r"(?<=\d)([.:])(?=\d)")

# Private-use placeholders: park the kept separators while the general punctuation
# pass runs, then put them back.
_PLACEHOLDERS = {".": "", ":": ""}
_RESTORE = {ord(v): k for k, v in _PLACEHOLDERS.items()}


def _punctuation_to_space() -> dict[int, str]:
    """Every Unicode punctuation codepoint (category P*) mapped to a space.

    A space, not nothing: "ரயில்,பஸ்" must not become one token.
    Marks (M*) are left alone -- the pulli and the vowel signs are the writing system.
    """
    table: dict[int, str] = {}
    for cp in range(0x110000):
        if unicodedata.category(chr(cp)).startswith("P"):
            table[cp] = " "
    return table


_PUNCTUATION = _punctuation_to_space()


def norm_text(text: str) -> str:
    """Return the canonical comparison form of ``text``."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = text.translate(_TAMIL_DIGITS)
    text = text.translate(_APOSTROPHES)
    text = _GROUPING_COMMA.sub("", text)
    text = _INTRA_NUMBER.sub(lambda m: _PLACEHOLDERS[m.group(1)], text)
    text = text.translate(_PUNCTUATION)
    text = text.translate(_RESTORE)
    text = text.lower()  # no-op for Tamil (uncased); folds code-mixed Latin
    text = _WHITESPACE.sub(" ", text)
    return text.strip()
