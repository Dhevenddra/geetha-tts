r"""Where, in the text, does a known stretch of audio sit?

The breath generator (D-057) writes a breath into a clip at a known time. A training row
needs the ``<|nv_breath|>`` tag at the same place in the *text*. This module aligns the
words to the audio (torchaudio's MMS forced aligner over romanised Tamil) and names the
word boundary a time span falls in -- or refuses, when the span sits inside a word.

Nothing here judges whether the span is a breath; that is ``geetha.eval.breath_score``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

__all__ = ["WordSpan", "Placement", "romanise_words", "boundary_for_span", "place_tags",
           "Aligner"]

_NOT_LETTER = re.compile(r"[^a-z]")


@dataclass(frozen=True)
class WordSpan:
    word: str
    start_s: float
    end_s: float
    score: float


@dataclass(frozen=True)
class Placement:
    """``before_word`` = index of the word the span comes before (0 = before everything)."""
    before_word: int
    gap_start_s: float
    gap_end_s: float
    covered: float          # share of the span that lies inside the gap between the words


def romanise_words(words: Sequence[str], romanise) -> list[str]:
    """Lower-case a-z only, one string per word; a word that romanises to nothing is ``""``."""
    return [_NOT_LETTER.sub("", romanise(word).lower()) for word in words]


def boundary_for_span(words: Sequence[WordSpan], from_s: float, to_s: float,
                      clip_s: float) -> Placement:
    """The word boundary whose silent gap overlaps ``[from_s, to_s]`` the most."""
    if to_s <= from_s:
        raise ValueError("empty span")
    best: Placement | None = None
    for index in range(len(words) + 1):
        gap_start = words[index - 1].end_s if index else 0.0
        gap_end = words[index].start_s if index < len(words) else clip_s
        overlap = max(0.0, min(to_s, gap_end) - max(from_s, gap_start))
        candidate = Placement(index, gap_start, gap_end, overlap / (to_s - from_s))
        if best is None or candidate.covered > best.covered:
            best = candidate
    assert best is not None
    return best


def place_tags(text_words: Sequence[str], before_words: Sequence[int], tag: str) -> str:
    """``text_words`` with ``tag`` written before each word index in ``before_words``."""
    marks = sorted(set(before_words))
    if marks and (marks[0] < 0 or marks[-1] > len(text_words)):
        raise ValueError(f"tag positions {marks} outside 0..{len(text_words)}")
    out: list[str] = []
    for index, word in enumerate(text_words):
        if index in marks:
            out.append(tag)
        out.append(word)
    if len(text_words) in marks:
        out.append(tag)
    return " ".join(out)


class Aligner:
    """MMS forced alignment. Heavy imports stay inside, so the pure functions test anywhere."""

    def __init__(self, device: str = "cpu") -> None:
        import torchaudio
        import uroman

        bundle = torchaudio.pipelines.MMS_FA
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.device = device
        self._roman = uroman.Uroman().romanize_string

    def words(self, wave, sample_rate: int, text_words: Sequence[str]) -> list[WordSpan]:
        """One :class:`WordSpan` per word of ``text_words``, in order."""
        import torch
        import torchaudio

        roman = romanise_words(text_words, self._roman)
        if any(not r for r in roman):
            raise ValueError("a word has no letters after romanisation")
        audio = torch.as_tensor(wave, dtype=torch.float32).reshape(1, -1)
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
        with torch.inference_mode():
            emission, _ = self.model(audio.to(self.device))
            spans = self.aligner(emission[0], self.tokenizer(roman))
        seconds_per_frame = audio.shape[1] / emission.shape[1] / self.sample_rate
        return [WordSpan(word, pieces[0].start * seconds_per_frame,
                         pieces[-1].end * seconds_per_frame,
                         sum(p.score * (p.end - p.start) for p in pieces)
                         / max(1, sum(p.end - p.start for p in pieces)))
                for word, pieces in zip(text_words, spans)]
