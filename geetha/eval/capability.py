r"""Naturalness-Ta, CAPABILITY component — the claim D-021 put at the front.

What is being claimed
---------------------
Indic-Speak's model card says: *"No sound effects. No laughter, sighs, or breaths."*
C2 measured on 27 Oct 2026 that this is true in the strongest possible sense -- the
sixteen non-verbal tokens the tokenizer holds are **inert**. Inserting
``<|nv_breath|>`` moved the audio 15.166 from the unmarked sentence against a noise
floor of 16.773: less than re-rolling the seed. So a Geetha that breathes is not a
better-sounding model, it is a model doing something the base cannot do at all.

Two things that are not the same thing
--------------------------------------
The re-spec wrote this component as "breath/filler events per minute". Reading the
token contract makes clear those are two different measurements:

* **non-verbal events** -- breath, sigh, throat-clear, laugh. Acoustic. Produced by the
  closed set of sixteen tokens, which C2 showed do nothing in the base model. The gap
  here is **binary**, and it is the headline.
* **fillers** -- "அதாவது", "உம்". There is no filler token (``contract.py``: *"fillers
  are speech, not events, and the closed set has no filler token"*), so a filler is
  Tamil text the model reads. **The base model can say a filler perfectly well.** What
  it probably cannot do is make one *sound* like hesitation rather than like a word
  being read. That is a difference of degree, it belongs in the report as such, and
  claiming it as part of the binary gap would be overclaiming.

So the prompt set keeps the two apart, and the published sentence is about non-verbal
events.

Why there is a clean control
----------------------------
Events per minute, on its own, rewards a model that wheezes through every sentence.
``clean`` items are fluent read-aloud Tamil where a breath mid-phrase is a **defect**,
and they give a false-positive rate. Without them the headline number is unfalsifiable,
which is the same failure mode D-020 removed from the realness test.

Prompts must fit one pass
-------------------------
``geetha.text.chunk`` exists because B7 truncated 145 of 262 prompts. A naturalness
prompt long enough to need chunking carries a pause the writer did not write, so these
prompts are validated to render in a single pass -- measured on the *spoken* text, since
a mark costs characters but no speaking time.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from geetha.text.chunk import split_for_synthesis
from geetha.text.contract import NONVERBAL_TOKENS

__all__ = [
    "CONDITIONS",
    "Condition",
    "EXPECTED_COUNTS",
    "KINDS",
    "MAX_PROMPT_CHARS",
    "SOURCE_PATH",
    "CapabilityItem",
    "event_rate_per_minute",
    "false_positive_rate",
    "load_source",
    "marks_in",
    "render_text",
    "spoken_text",
    "validate_items",
]

SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "naturalness_ta_source.tsv"
)

# Sized for the measurement, not for maximalism. C2 puts the base model's non-verbal
# rate at zero, so the event/clean contrast needs no large sample to separate; what it
# needs is a clean control big enough to put a believable bound on false positives.
# Every item costs the operator review time, and his Tamil review is already what gates
# M3, so 84 items is a deliberate ceiling.
EXPECTED_COUNTS = {"event": 36, "filler": 24, "clean": 24}

Kind = Literal["event", "filler", "clean"]
KINDS: tuple[str, ...] = ("event", "filler", "clean")

# Matches the shape of a non-verbal token, valid or not, so an invented one is caught
# rather than left in the text to be spoken out literally.
_MARK = re.compile(r"<\|nv_[a-z_]+\|>")

# Same budget as the B7 harness: 180 characters is ~21 s at the slowest rate measured
# for a prompt that long, inside a 22 s single pass.
MAX_PROMPT_CHARS = 180


@dataclass(frozen=True)
class Condition:
    """One way of presenting an item to a system."""

    name: str
    description: str
    for_baselines: bool


CONDITIONS: dict[str, Condition] = {
    "marked": Condition(
        name="marked",
        description="the item's own text, non-verbal tokens intact — Geetha only",
        for_baselines=False,
    ),
    "stripped": Condition(
        name="stripped",
        description="tokens removed; the condition every system can be given, "
                    "including the base model and the three paid APIs",
        for_baselines=True,
    ),
}


@dataclass(frozen=True)
class CapabilityItem:
    kind: str
    id: str
    text: str
    gloss: str
    expected_events: int


def marks_in(text: str) -> list[str]:
    """Every non-verbal-token-shaped substring, in reading order."""
    return _MARK.findall(text)


def spoken_text(text: str) -> str:
    """The text with marks removed — what is actually said, and what duration tracks.

    Whitespace left behind by a removed mark is closed up, including before
    punctuation: a mark sitting in front of a full stop would otherwise leave
    ``"வந்தேன் ."``, which both reads wrong and makes two placements of the same
    breath look like two different sentences.
    """
    without = re.sub(r"\s+", " ", _MARK.sub(" ", text)).strip()
    return re.sub(r"\s+(?=[.,;:!?…])", "", without)


def render_text(item: CapabilityItem, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}, expected one of {list(CONDITIONS)}")
    return item.text if condition == "marked" else spoken_text(item.text)


def validate_items(items: Sequence[CapabilityItem]) -> None:
    """Raise on anything that would weaken or falsify the capability measurement."""
    seen_ids: set[str] = set()
    seen_spoken: dict[str, str] = {}
    for item in items:
        if item.kind not in KINDS:
            raise ValueError(f"{item.id}: unknown kind {item.kind!r}, expected one of {KINDS}")
        if not item.id:
            raise ValueError("an item has no id")
        if item.id in seen_ids:
            raise ValueError(f"duplicate id {item.id!r}")
        seen_ids.add(item.id)
        if not item.gloss.strip():
            raise ValueError(f"{item.id}: no gloss — the operator cannot review it")

        marks = marks_in(item.text)
        unknown = [mark for mark in marks if mark not in NONVERBAL_TOKENS]
        if unknown:
            raise ValueError(
                f"{item.id}: {unknown} not in the model's token contract (D-017). "
                f"The tokenizer would read it as literal text and the voice would "
                f"say it out loud."
            )

        if item.kind == "clean":
            if marks:
                raise ValueError(
                    f"{item.id}: a clean item carries {marks}. Clean items are where an "
                    f"event is a defect; a mark here destroys the false-positive rate."
                )
            if item.expected_events != 0:
                raise ValueError(f"{item.id}: a clean item must expect 0 events")
        elif item.kind == "filler":
            if marks:
                raise ValueError(
                    f"{item.id}: a filler item carries {marks}. A filler is Tamil text "
                    f"the model speaks, not a token — the base model can say it too, so "
                    f"mixing the two would claim a gap that is not there."
                )
            if item.expected_events != 0:
                raise ValueError(
                    f"{item.id}: a filler item expects 0 non-verbal events "
                    f"(its hesitation is lexical)"
                )
        elif len(marks) != item.expected_events:
            raise ValueError(
                f"{item.id}: expected_events is {item.expected_events} but the text "
                f"carries {len(marks)} mark(s)"
            )

        spoken = spoken_text(item.text)
        if not spoken:
            raise ValueError(f"{item.id}: nothing is spoken")
        if spoken in seen_spoken:
            raise ValueError(
                f"{item.id}: duplicate spoken text, already used by "
                f"{seen_spoken[spoken]!r}"
            )
        seen_spoken[spoken] = item.id

        if len(split_for_synthesis(spoken, MAX_PROMPT_CHARS)) > 1:
            raise ValueError(
                f"{item.id}: {len(spoken)} spoken characters does not fit a single pass "
                f"of {MAX_PROMPT_CHARS}. Shorten it — a chunked naturalness prompt "
                f"carries a pause the writer did not write."
            )


def event_rate_per_minute(events: int, seconds: float) -> float:
    """Non-verbal events per minute of audio."""
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    if events < 0:
        raise ValueError(f"events must not be negative, got {events}")
    return 60.0 * events / seconds


def false_positive_rate(clean_with_events: int, clean_total: int) -> float:
    """Share of clean prompts in which the detector found an event."""
    if clean_total <= 0:
        raise ValueError(
            "the clean set is empty, so there is no false-positive rate and the "
            "capability number is unfalsifiable"
        )
    if not 0 <= clean_with_events <= clean_total:
        raise ValueError(
            f"clean_with_events {clean_with_events} outside 0..{clean_total}"
        )
    return clean_with_events / clean_total


def load_source(path: str | Path = SOURCE_PATH) -> list[CapabilityItem]:
    """Parse the authored source TSV, skipping comment lines.

    Comments carry the reasoning behind the three kinds and must survive in the file
    the operator reviews, so they are stripped here rather than removed there.
    """
    items: list[CapabilityItem] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="	"):
            kind = (row.get("kind") or "").strip()
            if not kind or kind.startswith("#"):
                continue
            items.append(
                CapabilityItem(
                    kind=kind,
                    id=(row.get("id") or "").strip(),
                    text=(row.get("text") or "").strip(),
                    gloss=(row.get("gloss") or "").strip(),
                    expected_events=int((row.get("expected_events") or "0").strip() or 0),
                )
            )
    return items
