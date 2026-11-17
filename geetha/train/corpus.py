r"""The training corpus for one stage: prompt variants, repeats, replay, held-out slice.

Every decision here answers something the pilot exposed (D-051).

**The prompt is assembled the way the base model's own ``inference.py`` assembles it.**
Piece by piece, with no tokenizer-added specials::

    <|start_of_human|><|begin_of_text|> [speaker\n] [style\n] [env\n] text <|eot_id|>
    <|end_of_human|><|start_of_ai|>

The pilot encoded the whole prompt with ``tokenizer.encode(prompt)``, whose default
prepends a second ``<|begin_of_text|>`` -- so every one of its training examples began
with a token the base model never puts there, and every evaluation prompt (built by
``inference.py``) lacked it. Measured 10 Nov over 3,761 real rows: 0 matched the base
layout with the default call, 3,761 matched with ``add_special_tokens=False``.
:func:`load_items` therefore takes an encoder that must add nothing, and refuses one
that does.

**Conditioning dropout.** The pilot always trained with a style block and an env block,
and was then evaluated with neither. A model should not fall apart when a caller omits
a label, and "no style given" should mean *read the emotion off the text* (D-004), not
*undefined*. So a share of examples drop the style block, and a share of **studio**
examples drop the env block -- studio only, so that an absent env means studio and never
the phone-derived sound.

**Non-verbal tag dropout.** For a share of the clips that carry ``<|nv_*|>`` tokens, the
text is shown with the tokens removed and the audio unchanged. The model then has to
decide for itself where the breath goes, which is what "natural by default" means. The
tagged form still dominates, so the tag keeps its meaning.

**Stage B replay.** Stage B is Anitha and Arun only, and neither has a single non-verbal
token (D-050). It is also the last thing the model sees, and sequential fine-tuning on
data without a behaviour is the textbook way to forget it. So a share of every Stage B
epoch is replayed Stage A clips that carry events, under their own speaker and their
own ``enhanced`` label. Nothing is added to the studio rows, which D-007 protects.

**A held-out slice, per source,** chosen by a hash of the clip's identity so it is the
same slice on every pod, every resume and both stages.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pyarrow.parquet as pq

from geetha.data.loader import LoaderStats
from geetha.data.sequence import TrainingSequence, training_sequence
from geetha.text.contract import SOURCE_LABELS

__all__ = ["Item", "Variant", "Corpus", "strip_nonverbal", "load_items", "is_held_out",
           "epoch_plan", "replay_pool", "FULL"]

EncodeText = Callable[[str], Sequence[int]]

_NONVERBAL = re.compile(r"<\|nv_[a-z_]+\|>")
_SPACES = re.compile(r"[ \t]+")

_HEAD = "<|start_of_human|><|begin_of_text|>"
_TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"
STUDIO = "studio"


def strip_nonverbal(text: str) -> str:
    """``text`` with every ``<|nv_*|>`` token removed and the gap closed."""
    return _SPACES.sub(" ", _NONVERBAL.sub(" ", text)).strip()


@dataclass(frozen=True)
class Variant:
    """Which form of a clip's prompt one training step sees."""
    untagged: bool = False     # non-verbal tokens removed from the text
    drop_style: bool = False
    drop_env: bool = False


FULL = Variant()


@dataclass(frozen=True)
class Item:
    key: str                         # source/shard/row -- stable across machines
    group: str                       # "ivr" or the studio speaker's name
    env: str
    head: tuple[int, ...]            # <|start_of_human|><|begin_of_text|> + speaker block
    style: tuple[int, ...]
    env_block: tuple[int, ...]
    text: tuple[int, ...]
    text_plain: tuple[int, ...] | None   # the same text with non-verbal tokens removed
    tail: tuple[int, ...]
    audio: tuple[int, ...]
    repeat: int = 1

    @property
    def has_events(self) -> bool:
        return self.text_plain is not None

    @property
    def length(self) -> int:
        """The longest form, plus the three tokens that frame the audio span."""
        return (len(self.head) + len(self.style) + len(self.env_block) + len(self.text)
                + len(self.tail) + len(self.audio) + 3)

    def prompt_ids(self, variant: Variant = FULL) -> list[int]:
        text = self.text_plain if (variant.untagged and self.text_plain is not None) \
            else self.text
        return [*self.head,
                *(() if variant.drop_style else self.style),
                *(() if variant.drop_env else self.env_block),
                *text, *self.tail]

    def sequence(self, variant: Variant = FULL) -> TrainingSequence:
        return training_sequence(self.prompt_ids(variant), self.audio)


@dataclass
class Corpus:
    train: list[Item] = field(default_factory=list)
    held_out: list[Item] = field(default_factory=list)


def is_held_out(key: str, *, per_thousand: int) -> bool:
    """Whether ``key`` falls in the held-out slice. Stable on every machine."""
    if per_thousand <= 0:
        return False
    digest = hashlib.sha256(f"geetha-heldout:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1000 < per_thousand


def load_items(
    paths: Iterable[Path | str],
    encode_text: EncodeText,
    *,
    max_length: int,
    stats: LoaderStats | None = None,
    held_out_per_thousand: int = 0,
) -> Corpus:
    """Read code shards into :class:`Item` s, split into train and held-out.

    ``encode_text`` must add **no** special tokens of its own. That is checked: the two
    halves of a string must encode to the same ids as the whole, which a tokenizer that
    prepends a BOS to every call cannot satisfy.
    """
    whole, left, right = (list(encode_text(s)) for s in ("ab cd", "ab", " cd"))
    if whole[0] == left[0] == right[0]:
        raise ValueError(
            "the text encoder adds a special token to every call (the first id of "
            f"every encoding is {whole[0]}); pass add_special_tokens=False. The pilot "
            "trained with a doubled <|begin_of_text|> for exactly this reason")

    def ids(text: str) -> tuple[int, ...]:
        return tuple(encode_text(text))

    head_ids, tail_ids = ids(_HEAD), ids(_TAIL)
    block_cache: dict[str, tuple[int, ...]] = {}

    def block(text: str) -> tuple[int, ...]:
        if text not in block_cache:
            block_cache[text] = ids(text)
        return block_cache[text]

    corpus = Corpus()
    for path in paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            if stats is not None:
                stats.rows += 1
            if row["env"] not in SOURCE_LABELS:
                raise ValueError(f"env {row['env']!r} is not one of {SOURCE_LABELS!r}")
            text = row["text"]
            plain = strip_nonverbal(text)
            speaker = row["speaker"]
            item = Item(
                key=f"{row['source']}/{row['shard']}/{row['row_index']}",
                group="ivr" if speaker.startswith("ivr") else speaker,
                env=row["env"],
                head=head_ids + (block(f"<|speaker>{speaker}<speaker|>\n") if speaker else ()),
                style=block(f"<|style>{row['style']}<style|>\n") if row["style"] else (),
                env_block=block(f"<|env>{row['env']}<env|>\n"),
                text=ids(text),
                text_plain=ids(plain) if (plain != text and plain) else None,
                tail=tail_ids,
                audio=tuple(row["tokens"]),
                repeat=max(1, row.get("repeat") or 1),
            )
            try:
                # Validates the audio span once, here, rather than on every step.
                training_sequence(item.prompt_ids(), item.audio, max_length=max_length)
            except ValueError as problem:
                if stats is not None:
                    if "context" in str(problem):
                        stats.skipped_too_long += 1
                    else:
                        stats.skipped_malformed += 1
                continue
            if stats is not None:
                stats.examples += 1
            if is_held_out(item.key, per_thousand=held_out_per_thousand):
                corpus.held_out.append(item)
            else:
                corpus.train.append(item)
    return corpus


def replay_pool(items: Sequence[Item]) -> list[int]:
    """Indices of the clips worth replaying in Stage B: the ones that carry events."""
    return [index for index, item in enumerate(items) if item.has_events]


def epoch_plan(
    items: Sequence[Item],
    *,
    seed: int,
    epoch: int,
    honour_repeats: bool,
    nv_dropout: float,
    style_dropout: float = 0.0,
    env_dropout: float = 0.0,
    replay_from: Sequence[int] = (),
    replay_fraction: float = 0.0,
) -> tuple[list[int], dict[int, Variant]]:
    """Which examples this epoch uses, and which prompt form each one is shown in.

    Returns ``(indices, variants)``. ``indices`` may repeat an index (Stage B's
    upweighting); an index absent from ``variants`` is shown in full. ``replay_from``
    are indices into ``items`` that are not part of the stage proper and are sampled so
    that they make up ``replay_fraction`` of the epoch.
    """
    for name, value in (("nv_dropout", nv_dropout), ("style_dropout", style_dropout),
                        ("env_dropout", env_dropout), ("replay_fraction", replay_fraction)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be within [0, 1), got {value}")

    rng = random.Random(f"geetha-epoch:{seed}:{epoch}")
    replay_set = set(replay_from)
    indices: list[int] = []
    for index, item in enumerate(items):
        if index in replay_set:
            continue
        indices.extend([index] * (item.repeat if honour_repeats else 1))

    if replay_from and replay_fraction > 0.0:
        wanted = round(len(indices) * replay_fraction / (1.0 - replay_fraction))
        pool = sorted(replay_set)
        indices.extend(pool if wanted >= len(pool) else rng.sample(pool, wanted))

    variants: dict[int, Variant] = {}
    for index in sorted(set(indices)):
        item = items[index]
        # Three draws per clip whether or not they apply, so changing one rate does not
        # reshuffle the other two.
        rolls = (rng.random(), rng.random(), rng.random())
        variant = Variant(
            untagged=item.has_events and rolls[0] < nv_dropout,
            drop_style=bool(item.style) and rolls[1] < style_dropout,
            # Studio only: an absent env must always mean studio, never phone-derived.
            drop_env=item.env == STUDIO and rolls[2] < env_dropout,
        )
        if variant != FULL:
            variants[index] = variant
    return indices, variants
