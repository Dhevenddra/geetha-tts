"""The inline direction-tag inventory for Geetha-TTS v1 (runbook sections 1 and 8).

**The inventory comes from the model, not from the runbook.** See
``geetha.text.contract``: the runbook invented ``<angry>`` and ``<neutral>``, but
Indic-Speak was trained on ``anger`` and has no ``neutral`` label. Conditioning on a
label the model never saw is silent misconditioning, so an unknown emotion raises here.

Two shapes exist on purpose:

* ``<emotion>`` -- angle brackets, one per utterance, carried in the style field;
* ``[nonverbal]`` -- square brackets, inline, from IndicVoices verbatim transcripts;
  ``[emphasis] ... [/emphasis]`` wraps the words it emphasises.

``[laugh]`` is listed but ships only if it measurably works on Tamil (runbook
section 1); keeping it in the inventory means training data can carry it while the
release decision stays open.

This module handles the *bracketed source notation* ETL step 4 writes. Converting that
notation to the model's own token ids is ``contract.map_corpus_nonverbal``.
"""

from __future__ import annotations

import re

from geetha.text.contract import EMOTION_STYLES

__all__ = ["EMOTION_TAGS", "NONVERBAL_TAGS", "SPAN_TAGS", "strip_tags"]

EMOTION_TAGS = frozenset(EMOTION_STYLES)
NONVERBAL_TAGS = frozenset({"breath", "uh", "um", "tsk", "laugh"})
SPAN_TAGS = frozenset({"emphasis"})

_KNOWN_SQUARE = NONVERBAL_TAGS | SPAN_TAGS | {f"/{name}" for name in SPAN_TAGS}
_BRACKETED = re.compile(r"\[[^\[\]]*\]|<[^<>]*>")
_WHITESPACE = re.compile(r"\s+")


def _is_known(token: str) -> bool:
    inner = token[1:-1].strip()
    if token.startswith("<"):
        # Our inline notation is lowercase (<happy>); the model's style label is
        # uppercase (HAPPY). Different layers, so compare case-insensitively -- and
        # <angry> still fails, because it is not a trained label in any casing.
        return inner.upper() in {tag.upper() for tag in EMOTION_TAGS}
    return inner in _KNOWN_SQUARE


def strip_tags(text: str, *, strict: bool = True) -> str:
    """Remove every known direction tag, keeping the words they wrap.

    With ``strict`` (the default) an unrecognised bracketed token raises
    ``ValueError``: a tag silently deleted from a reference becomes an invisible
    error in a published CER number.
    """
    unknown = [m.group(0) for m in _BRACKETED.finditer(text) if not _is_known(m.group(0))]
    if unknown and strict:
        raise ValueError(f"unknown inline tag(s) {unknown} in {text!r}")

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return " " if _is_known(token) else token

    stripped = _BRACKETED.sub(replace, text)
    return _WHITESPACE.sub(" ", stripped).strip()
