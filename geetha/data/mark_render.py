r"""Turn IndicVoices marked text into TTS training text -- ETL step 4 (D-035).

Each kind of mark has a different fate, decided in D-035 and summarised here:

====================  =========================================================
mark                  becomes
====================  =========================================================
``breathing``         ``<|nv_breath|>`` and the other 16 non-verbal tokens, via
``stammers``          ``geetha.text.contract.map_corpus_nonverbal``
``laughter`` ...
``uhh`` / ``umm``     Tamil text -- ``ஆ`` / ``ம்``. A filler is *speech*, and the
``hmm``               closed token set has no filler slot. Both spellings are
                      the corpus's own, attested 3,189 and 890 times where a
                      transcriber wrote the sound instead of tagging it
``tsk``               nothing. A dental click is none of the 16 tokens, and
                      mapping it to the nearest would mislabel 1,068 rows
noise, horn, bell,    nothing -- environment, not voice
``persistent-*`` ...
``unintelligible``    **the row is rejected.** These mean the audio contains
``talking``           speech the text does not account for, sometimes another
``child_talking``     person's. Training on them teaches the model to utter
                      words nobody asked for -- a text/audio mismatch, not noise
====================  =========================================================

Punctuation is never touched. ``voices.md`` says the model derives pauses and
intonation from punctuation, so it is load-bearing in training text: the transcriber
already put commas where the speaker paused, and that is signal. Only runs of
whitespace left behind by a removed mark are collapsed.
"""

from __future__ import annotations

import re

from geetha.text.contract import map_corpus_nonverbal

__all__ = ["FILLER_TEXT", "REJECTING_MARKS", "RejectedRow", "render_training_text"]

# Tamil spellings taken from IndicVoices' own transcription, not invented: the
# lengthened vowel and the bare nasal. Counted over 12 shards: ஆ 3,189, ம் 890.
FILLER_TEXT = {
    "uhh": "ஆ",
    "uh": "ஆ",
    "umm": "ம்",
    "um": "ம்",
    "hmm": "ம்",
}

# Marks whose presence means the audio holds speech the text does not.
REJECTING_MARKS = frozenset({"unintelligible", "talking", "child_talking"})

# Environment and click marks, named so the *unclosed*-mark pass below can recognise a
# fragment as a mark rather than as Tamil text. The closed-mark pass does not need this
# list: there, anything whose name is unknown is removed.
DISCARDED_MARKS = frozenset({
    "noise", "persistent-noise-start", "persistent-noise-end", "bird_squawk",
    "horn", "bell", "tsk",
})

_MARK = re.compile(r"[\[<]([^\[\]<>]{1,40})[\]>]")
# Any bracket opening followed by a mark-shaped name, closed or not. Used to harvest
# names for the rejection check and to rescue unclosed marks. No spaces in the name:
# allowing them lets a greedy match swallow the Tamil or English word that follows.
_OPENING = re.compile(r"[\[<]\s*([A-Za-z][A-Za-z0-9_\-]{0,38})")
# A stray bracket character that is not part of one of the model's own <|...|> tokens.
_STRAY_SQUARE = re.compile(r"[\[\]]")
_STRAY_OPEN = re.compile(r"<(?!\|)")
_STRAY_CLOSE = re.compile(r"(?<!\|)>")
_WHITESPACE = re.compile(r"\s+")


class RejectedRow(Exception):
    """This row must not enter a training manifest. The message names why."""


def _canonical(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def render_training_text(marked_text: str | None) -> str:
    """Training text for one clip, or raise :class:`RejectedRow`.

    Raises rather than returning a sentinel because a rejected row is not a value a
    caller should be able to forget to check.
    """
    source = marked_text or ""

    # Rejection is decided before anything is rendered, so a row cannot be partly
    # salvaged into training by the good marks it also happens to carry. Names are
    # harvested from every bracket opening, closed or not, because the corpus contains
    # nested marks like "[talking [noise]]" whose outer name has no closing bracket.
    for raw in _OPENING.findall(source):
        name = _canonical(raw)
        if name in REJECTING_MARKS:
            raise RejectedRow(f"{name}: the audio contains speech the text does not account for")

    def rendered_mark(name: str) -> str | None:
        """The text a mark becomes, or ``None`` if the name means nothing to us."""
        if name in FILLER_TEXT:
            return FILLER_TEXT[name]
        token = map_corpus_nonverbal(name)
        if token is not None:
            return token
        if name in DISCARDED_MARKS:
            return " "
        return None

    def replace_closed(match: re.Match[str]) -> str:
        # A well-formed mark whose name we do not recognise is still a mark, so it is
        # removed rather than passed through as literal brackets.
        return rendered_mark(_canonical(match.group(1))) or " "

    def replace_unclosed(match: re.Match[str]) -> str:
        # Here the brackets are broken, so an unrecognised name could be ordinary text.
        # Only a name we actually know is rewritten; anything else keeps its characters
        # and loses only the stray bracket, in the cleanup below.
        replacement = rendered_mark(_canonical(match.group(1)))
        return replacement if replacement is not None else match.group(1)

    text = _MARK.sub(replace_closed, source)
    text = _OPENING.sub(replace_unclosed, text)
    text = _STRAY_CLOSE.sub(" ", _STRAY_OPEN.sub(" ", _STRAY_SQUARE.sub(" ", text)))

    rendered = _WHITESPACE.sub(" ", text).strip()
    if not rendered:
        raise RejectedRow("no_text: nothing but marks")
    return rendered
