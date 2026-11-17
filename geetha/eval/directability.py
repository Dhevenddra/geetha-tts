"""Building the Directability-Ta prompt set (task B4, runbook §6).

Three kinds of item, each measuring something the others cannot:

* **carrier** — emotion-neutral text, rendered under every emotion label. Because the
  *same* text is rendered under different tags, any difference in delivery is
  attributable to the tag rather than to the words. Measuring directability on
  emotionally loaded text would mostly measure the model's reading of the text.
* **emotional** — text whose content implies a feeling, rendered with **no** tag. This
  is D-004: tags steer, but untagged text must also sound right.
* **probe** — a short phrase whose intent changes with delivery ("பாத்துக்கலாம்" can
  reassure or threaten). Listeners pick from a fixed intent list, so chance is
  ``1/len(intents)`` and is recorded per probe rather than assumed to be 25%.

Source text lives in ``geetha/data/directability_ta_source.tsv``, version-controlled and
reviewed by the operator before the eval vault is frozen. The Tamil was authored by
AI assistance and **needs that sign-off**: a sentence that reads oddly to a Tamil speaker
makes the measurement suspect, and no amount of downstream rigour repairs it.

Note the deliberate asymmetry with ``geetha.text.normalise``: prompts keep their
punctuation, because ``voices.md`` says the model derives pauses and intonation from it.
``norm_text`` is used here only to hash for leak detection, never to rewrite a prompt.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from geetha.runlog import text_hash
from geetha.text.contract import is_known_style
from geetha.text.normalise import norm_text

__all__ = [
    "CARRIER_STYLES",
    "SOURCE_PATH",
    "SourceItem",
    "build_items",
    "estimate_seconds",
    "find_leaks",
    "load_source",
    "validate_source",
]

SOURCE_PATH = Path(__file__).resolve().parent.parent / "data" / "directability_ta_source.tsv"

# 40 carriers x 5 styles = the 200 tagged prompts runbook B4 asks for. DISGUST is left
# out of the carriers — "disgust" over "the book is on the table" is the weakest of the
# six — and is covered instead by the untagged emotional items, which include
# disgust-laden content. Stated rather than silently dropped.
CARRIER_STYLES = ("HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE")

KINDS = ("carrier", "emotional", "probe")
EXPECTED_COUNTS = {"carrier": 40, "emotional": 60, "probe": 40}

# Clip-split threshold from runbook §8: 20 s at 82 tokens/s sits inside the 2,048-token
# sequence budget with room for the text prompt.
MAX_PROMPT_SECONDS = 20.0


@dataclass(frozen=True)
class SourceItem:
    kind: str
    id: str
    text: str
    gloss: str
    intents: tuple[str, ...]


def load_source(path: str | Path = SOURCE_PATH) -> list[SourceItem]:
    """Parse the authored source TSV, skipping comment lines."""
    items: list[SourceItem] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            kind = (row.get("kind") or "").strip()
            if not kind or kind.startswith("#"):
                continue
            intents = tuple(
                part.strip() for part in (row.get("intents") or "").split("|") if part.strip()
            )
            items.append(
                SourceItem(
                    kind=kind,
                    id=(row.get("id") or "").strip(),
                    text=(row.get("text") or "").strip(),
                    gloss=(row.get("gloss") or "").strip(),
                    intents=intents,
                )
            )
    return items


def estimate_seconds(text: str, *, chars_per_second: float) -> float:
    """Spoken duration estimated from a voice's measured rate.

    ``voices.md`` gives each voice a rate (Anitha 13.2 chars/sec, Arun 14.5) and says to
    plan script length against it. Use the *slower* voice to bound a prompt set.
    """
    return len(text) / chars_per_second


def validate_source(items: Sequence[SourceItem]) -> None:
    """Raise on anything that would quietly weaken the measurement."""
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    for item in items:
        if item.kind not in KINDS:
            raise ValueError(f"{item.id}: unknown kind {item.kind!r}, expected one of {KINDS}")
        if not item.id:
            raise ValueError(f"an item of kind {item.kind!r} has no id")
        if item.id in seen_ids:
            raise ValueError(f"duplicate id {item.id!r}")
        seen_ids.add(item.id)
        if not item.text.strip():
            raise ValueError(f"{item.id}: empty text")
        if not item.gloss.strip():
            raise ValueError(f"{item.id}: no gloss — the operator cannot review it")

        key = norm_text(item.text)
        if key in seen_text:
            raise ValueError(
                f"{item.id} and {seen_text[key]} have the same normalised text; two "
                f"prompts that are really one would silently halve a comparison"
            )
        seen_text[key] = item.id

        if item.kind == "probe":
            if len(item.intents) < 2:
                raise ValueError(
                    f"{item.id}: a probe needs at least two intents to choose between"
                )
        elif item.intents:
            raise ValueError(f"{item.id}: only probes declare intents")

        if estimate_seconds(item.text, chars_per_second=13.2) >= MAX_PROMPT_SECONDS:
            raise ValueError(
                f"{item.id}: about {estimate_seconds(item.text, chars_per_second=13.2):.1f} s "
                f"at Anitha's rate, over the {MAX_PROMPT_SECONDS:g} s clip budget"
            )

    for style in CARRIER_STYLES:
        if not is_known_style(style):
            raise ValueError(f"carrier style {style!r} is not a documented trained label")


def build_items(items: Sequence[SourceItem]) -> list[dict]:
    """Expand the source into the prompt set actually evaluated."""
    built: list[dict] = []

    for item in items:
        if item.kind != "carrier":
            continue
        for style in CARRIER_STYLES:
            built.append(
                {
                    "id": f"{item.id}/{style}",
                    "kind": "tagged",
                    "carrier_id": item.id,
                    "text": item.text,
                    "gloss": item.gloss,
                    "style": style,
                    "intents": None,
                    "chance": None,
                }
            )

    for item in items:
        if item.kind != "emotional":
            continue
        built.append(
            {
                "id": item.id,
                "kind": "untagged_emotional",
                "carrier_id": None,
                "text": item.text,
                "gloss": item.gloss,
                "style": None,
                "intents": None,
                "chance": None,
            }
        )

    for item in items:
        if item.kind != "probe":
            continue
        built.append(
            {
                "id": item.id,
                "kind": "probe",
                "carrier_id": None,
                "text": item.text,
                "gloss": item.gloss,
                "style": None,
                "intents": list(item.intents),
                "chance": 1 / len(item.intents),
            }
        )

    return built


def find_leaks(items: Iterable[SourceItem], training_text: set[str]) -> list[tuple[str, str]]:
    """Items whose normalised text appears in training (runbook rule 3).

    ``training_text`` holds already-normalised sentences, so the comparison is the same
    leak hash the ETL uses. Returns ``(id, text)`` for each collision, in source order.
    """
    hashes = {text_hash(t) for t in training_text}
    return [(item.id, item.text) for item in items if text_hash(item.text) in hashes]
