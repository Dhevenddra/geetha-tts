r"""Geetha: Tamil text to speech in two studio voices, with style direction.

    from inference import Geetha
    tts = Geetha("PATH_OR_REPO_OF_THE_GEETHA_ADAPTER")
    wav = tts.speak("நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", speaker="anitha", style="HAPPY")
    tts.save("out.wav", wav)                      # float32 mono, 24 kHz

Geetha is a LoRA adapter on ``bodhan-ai/indic-speak`` (Orpheus-style: a language model
writes SNAC codec tokens, a Vocos decoder turns them into audio). This file is the whole
inference recipe, because three things in it are not optional:

  * the PROMPT LAYOUT the adapter was trained on - speaker, style and ``env=studio``
    blocks, speaker names in lower case, no tokenizer-added BOS. A different layout
    still produces speech, of the base model's quality;
  * the DECODING - temperature 0.6 for neutral speech; for expressive styles
    temperature 0.9, repetition penalty 1.1 and classifier-free guidance on the style
    block (two forward passes per step);
  * the VALIDITY GATE - a clip that is silent or never reached end-of-speech is thrown
    away and re-rolled, at most ``retries`` times. Benchmarks report single-shot numbers
    (``retries=0``) first.

Two conveniences sit on top:

  * NUMBERS - digits, dates, times, rupees and percentages are rewritten as Tamil words
    before synthesis (``tamil_text.py``; ``numbers=False`` turns it off);
  * BREATH - write ``[breath]`` in the text and one audible breath is put into that
    sentence. The model does not produce a breath on request in these two voices, so
    this is done by hand: a breath the model itself made (``breaths.json``, codec
    tokens) is written into the token stream and the model carries on from it. Only a
    tag at the very START of a sentence is honoured by default: a breath before the
    first word, which listeners accepted as the speaker's own. A tag in the middle is
    ignored unless ``mid_breath=True``; that breath lands at the first pause the speaker
    makes, often not where the tag was written, and can sound pasted in. One breath per
    sentence at most.

The adapter is attached UNMERGED: the base model ties its input and output embeddings
and the adapter trains them, so merging changes what the model computes.

Needs: torch, transformers>=5, peft, snac, soundfile, and access to the gated base model.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000
BASE_MODEL = "bodhan-ai/indic-speak"
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
TOKENS_PER_SECOND = 82
VOICES = ("anitha", "arun")
EMOTIONS = ("HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST")
NEUTRAL_STYLE = "CONV"
SILENCE_DBFS = -45.0          # a clip quieter than this overall is mute
MAX_CHARS_PER_CHUNK = 220     # training clips are at most ~20 s; longer text is split

BREATH_TAG = "<|nv_breath|>"  # the model's own token; "[breath]" is accepted and rewritten
FRAME_TOKENS = 7              # one codec frame, about 85 ms
# When to write a mid-sentence breath: the speaker has just gone quiet (newest frame at
# or under SILENT_DB) after speaking (a frame over REARM_DB), at least 4 frames in.
SILENT_DB, REARM_DB = -42.0, -35.0
CONTEXT_FRAMES = 8            # frames decoded to measure the newest one

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from tamil_text import verbalise  # noqa: E402  (shipped next to this file)

_BREATH_WRITTEN = re.compile(r"\[\s*breath\s*\]", re.IGNORECASE)


def load_breaths(path: str | Path | None = None) -> dict[str, list[list[int]]]:
    """``{voice: [breath as a flat list of codec tokens, ...]}`` from breaths.json."""
    path = Path(path) if path else _HERE / "breaths.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[list[int]]] = {}
    for breath in data["breaths"]:
        if breath["tokens"] and len(breath["tokens"]) % FRAME_TOKENS == 0:
            out.setdefault(breath["voice"], []).append(list(breath["tokens"]))
    return out


def breath_plan(chunk: str) -> tuple[str, str | None]:
    """(text to send, where the breath goes: 'lead', 'mid' or None)."""
    chunk = " ".join(_BREATH_WRITTEN.sub(f" {BREATH_TAG} ", chunk).split())
    if BREATH_TAG not in chunk:
        return chunk, None
    return chunk, "lead" if chunk.startswith(BREATH_TAG) else "mid"


def split_at_breath(chunk: str) -> list[tuple[str, str | None, bool]]:
    """A chunk with a mid-sentence tag, cut AT the tag: [(text, place, joins_previous)].

    The words before the tag are spoken as one piece; the words after it as a second
    piece that opens with a lead breath, the placement listeners accepted. The breath
    then sits exactly where it was written. Only the first tag is honoured.
    """
    head, _, tail = chunk.partition(BREATH_TAG)
    head, tail = head.strip(), " ".join(tail.replace(BREATH_TAG, " ").split())
    if not head or not tail:
        return [(chunk, None, False)]
    return [(head, None, False), (f"{BREATH_TAG} {tail}", "lead", True)]


_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True)
class Decoding:
    """How tokens are sampled. The defaults are the measured ones (see the model card)."""

    # Expressive styles: hotter sampling and a repetition penalty, with style guidance.
    temperature: float = 0.9
    repetition_penalty: float = 1.1
    # Neutral speech: measured on 41 prompts, the hotter setting costs 2.4-3.2 CER points
    # and buys nothing when no emotion is asked for.
    neutral_temperature: float = 0.6
    neutral_repetition_penalty: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    # Guidance scale on the style block, per style; 0 = off (one forward pass).
    # Guidance sharpens an emotion and costs intelligibility, so neutral speech has none.
    style_guidance: dict = field(default_factory=lambda: {NEUTRAL_STYLE: 0.0,
                                                          **{e: 3.5 for e in EMOTIONS}})
    default_guidance: float = 0.0

    def guidance_for(self, style: str) -> float:
        return float(self.style_guidance.get(style, self.default_guidance))

    def sampling_for(self, style: str) -> tuple[float, float]:
        """(temperature, repetition penalty) for this style."""
        if self.guidance_for(style) > 0:
            return self.temperature, self.repetition_penalty
        return self.neutral_temperature, self.neutral_repetition_penalty


def normalise_speaker(name: str) -> str:
    """Training rows say ``anitha`` / ``arun``; ``Anitha`` tokenises differently."""
    speaker = name.strip().lower()
    if speaker not in VOICES:
        raise ValueError(f"speaker must be one of {VOICES}, got {name!r}")
    return speaker


def build_prompt(text: str, speaker: str, style: str = NEUTRAL_STYLE) -> str:
    """The trained layout. ``style=''`` gives the unconditional branch used for guidance."""
    blocks = [f"<|speaker>{normalise_speaker(speaker)}<speaker|>\n"]
    if style:
        blocks.append(f"<|style>{style}<style|>\n")
    blocks.append("<|env>studio<env|>\n")
    return ("<|start_of_human|><|begin_of_text|>" + "".join(blocks) + text
            + "<|eot_id|><|end_of_human|><|start_of_ai|><|start_of_speech|>")


def split_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """Sentences, then clauses, then words: never cut inside a word or an inline tag."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []

    def add(piece: str) -> None:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)

    for sentence in _SENTENCE_BREAK.split(text):
        if len(sentence) <= max_chars:
            add(sentence)
            continue
        for clause in _CLAUSE_BREAK.split(sentence):
            if len(clause) <= max_chars:
                add(clause)
                continue
            line = ""
            for word in clause.split(" "):
                if line and len(line) + 1 + len(word) > max_chars:
                    chunks.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                chunks.append(line)
    return chunks


def is_valid(wav: np.ndarray, ended: bool) -> bool:
    """Mute clips and clips that never reached end-of-speech are not speech."""
    if not ended or len(wav) < SAMPLE_RATE // 10:
        return False
    level = 20 * np.log10(float(np.sqrt(np.mean(np.square(wav, dtype=np.float64)))) + 1e-9)
    return level > SILENCE_DBFS


class Geetha:
    """Loads the base model once, attaches the adapter, speaks on call."""

    sample_rate = SAMPLE_RATE

    def __init__(self, adapter: str, *, base: str = BASE_MODEL,
                 base_revision: str | None = BASE_REVISION, device: str | None = None,
                 token: str | None = None, breaths: str | Path | None = None):
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel

        local = base if Path(base).is_dir() else snapshot_download(
            base, revision=base_revision, token=token)
        # The base repo's own loader is also a file called inference.py. Loaded by path
        # under another name, or "from inference import TTS" would find THIS file.
        import importlib.util

        sys.path.insert(0, local)
        spec = importlib.util.spec_from_file_location(
            "indic_speak_inference", str(Path(local) / "inference.py"))
        loader = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loader
        spec.loader.exec_module(loader)
        TTS, ids_to_codes = loader.TTS, loader.ids_to_codes

        adapter_dir = adapter if Path(adapter).is_dir() else snapshot_download(
            adapter, token=token, allow_patterns=["adapter_config.json", "adapter_model.*"])
        self._tts = TTS(local, device=device)
        self._tts.lm = PeftModel.from_pretrained(self._tts.lm, adapter_dir).eval()
        if (self._tts.device.startswith("cuda")
                and torch.cuda.get_device_capability(0)[0] < 8):
            self._tts.lm = self._tts.lm.half()        # no native bf16 on older cards
        self._ids_to_codes = ids_to_codes
        self._tok = self._tts.tok
        self._stop = [self._tts._eos, self._tok.eos_token_id]
        self._breaths = load_breaths(breaths)

    # ------------------------------------------------------------------ one chunk
    def _encode(self, prompt: str) -> list[int]:
        return self._tok.encode(prompt, add_special_tokens=False)   # no BOS: as trained

    def _decode(self, tokens: list[int]) -> np.ndarray:
        import torch

        tts = self._tts
        try:
            with torch.no_grad():
                codes = self._ids_to_codes(tokens, tts._base, tts.device)
                wav = tts.vocos(tts.snac.quantizer.from_codes(codes).float())
            return np.asarray(wav[0, 0].clamp(-1, 1).float().cpu().numpy(),
                              dtype=np.float32).reshape(-1)
        except ValueError:                           # not one complete codec frame
            return np.zeros(0, dtype=np.float32)

    def _newest_frame_db(self, tokens: list[int]) -> float:
        """Level of the frame just completed, decoded with a little left context."""
        frames = min(CONTEXT_FRAMES, len(tokens) // FRAME_TOKENS)
        wav = self._decode(tokens[-frames * FRAME_TOKENS:]) if frames else np.zeros(0)
        if not len(wav):
            return 0.0
        tail = wav[-(len(wav) // frames):]
        return float(20 * np.log10(np.sqrt(np.mean(np.square(tail, dtype=np.float64))) + 1e-7))

    def _generate(self, text: str, speaker: str, style: str, decoding: Decoding,
                  seed: int | None, max_seconds: float, breath: list[int] | None = None,
                  place: str | None = None) -> tuple[np.ndarray, bool]:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        tts, scale = self._tts, decoding.guidance_for(style)
        temperature, repetition = decoding.sampling_for(style)
        ids = self._encode(build_prompt(text, speaker, style))

        class Guidance(LogitsProcessor):
            """Classifier-free guidance against the same prompt without its style block."""

            def __init__(self, other: list[int]):
                self.other = torch.tensor([other], device=tts.device)
                self.past, self.started = None, False

            def __call__(self, input_ids, scores):
                feed = input_ids[:, -1:] if self.started else self.other
                out = tts.lm(input_ids=feed, past_key_values=self.past, use_cache=True)
                self.past, self.started = out.past_key_values, True
                cond = torch.log_softmax(scores.float(), dim=-1)
                unc = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
                return (unc + scale * (cond - unc)).to(scores.dtype)

        newest_frame_db = self._newest_frame_db

        class BreathWriter(LogitsProcessor):
            """Writes one breath into the stream, token by token, then steps aside."""

            def __init__(self, tokens: list[int], lead: bool):
                self.tokens = list(tokens)
                self.pending = list(tokens) if lead else []
                self.used, self.armed = lead, True

            def __call__(self, input_ids, scores):
                made = input_ids.shape[1] - len(ids)
                if (not self.used and made and made % FRAME_TOKENS == 0):
                    level = newest_frame_db(input_ids[0, len(ids):].tolist())
                    if level > REARM_DB:
                        self.armed = True
                    elif level <= SILENT_DB and self.armed and made // FRAME_TOKENS >= 4:
                        self.pending, self.used = list(self.tokens), True
                if not self.pending:
                    return scores
                forced = torch.full_like(scores, float("-inf"))
                forced[:, self.pending.pop(0)] = 0.0
                return forced

        processors = LogitsProcessorList()
        if scale > 0 and style:
            processors.append(Guidance(self._encode(build_prompt(text, speaker, ""))))
        if breath and place in ("lead", "mid"):
            processors.append(BreathWriter(breath, lead=place == "lead"))   # must be last
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.tensor([ids], device=tts.device)
        with torch.no_grad():
            out = tts.lm.generate(
                input_ids=x, attention_mask=torch.ones_like(x),
                max_new_tokens=int(max_seconds * TOKENS_PER_SECOND), do_sample=True,
                temperature=temperature, top_p=decoding.top_p,
                top_k=decoding.top_k, repetition_penalty=repetition,
                eos_token_id=self._stop, pad_token_id=self._tok.eos_token_id,
                logits_processor=processors)
        new = out[0].tolist()[len(ids):]
        ended = bool(new) and new[-1] in self._stop
        return self._decode(new), ended

    # ------------------------------------------------------------------ public
    def speak(self, text: str, speaker: str = "anitha", style: str = NEUTRAL_STYLE, *,
              decoding: Decoding | None = None, seed: int | None = None,
              retries: int = 2, max_seconds: float = 22.0,
              pause_seconds: float = 0.25, numbers: bool = True,
              breath: bool = True, mid_breath: bool | str = False,
              join_seconds: float = 0.05) -> np.ndarray:
        """Synthesize ``text``. Long text is split and the pieces joined with a short pause.

        ``retries=0`` is single-shot, which is what the published benchmarks report first.
        ``numbers`` reads digits as Tamil words; ``breath`` honours a ``[breath]`` tag at
        the start of a sentence; ``mid_breath`` also honours one in the middle, and is
        experimental either way: ``True`` puts the breath at the speaker's first pause
        (often not where the tag is); ``"split"`` cuts the sentence at the tag and opens
        the second half with the breath (exactly at the tag; the two halves are separate
        takes joined ``join_seconds`` apart).
        """
        decoding = decoding or Decoding()
        speaker = normalise_speaker(speaker)
        if numbers:
            text = verbalise(text)
        library = self._breaths.get(speaker, []) if breath else []
        pieces: list[np.ndarray] = []
        gap = np.zeros(int(pause_seconds * SAMPLE_RATE), dtype=np.float32)
        units: list[tuple[str, str | None, bool]] = []
        for chunk in split_text(text):
            chunk, place = breath_plan(chunk)
            if place == "mid" and mid_breath == "split" and library:
                units.extend(split_at_breath(chunk))
            else:
                units.append((chunk, place if place != "mid" or mid_breath else None, False))
        join = np.zeros(int(join_seconds * SAMPLE_RATE), dtype=np.float32)
        for index, (chunk, place, joins_previous) in enumerate(units):
            wav, ended = np.zeros(0, dtype=np.float32), False
            for attempt in range(retries + 1):
                chunk_seed = None if seed is None else seed + 1000 * index + attempt
                pick = (np.random.randint(len(library)) if seed is None
                        else (seed + index + attempt) % len(library)) if library else 0
                wav, ended = self._generate(chunk, speaker, style, decoding, chunk_seed,
                                            max_seconds,
                                            breath=library[pick] if library and place else None,
                                            place=place)
                if is_valid(wav, ended):
                    break
            else:
                warnings.warn(f"chunk {index} stayed invalid after {retries + 1} attempts "
                              "(silent, or never reached end-of-speech)", RuntimeWarning,
                              stacklevel=2)
            if pieces:
                pieces.append(join if joins_previous else gap)
            pieces.append(wav)
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    @staticmethod
    def save(path: str, wav: np.ndarray) -> str:
        import soundfile as sf

        sf.write(path, wav, SAMPLE_RATE)
        return path
