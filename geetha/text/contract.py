"""Indic-Speak's own token contract — read from the model, not from the runbook.

Source: ``bodhan-ai/indic-speak`` — ``token_contract.md``, ``config.json`` and
``voices.md``, read 26 Oct 2026.

The runbook (§8, §1) specified a tag scheme of our own invention: new tokens
``[breath] [uh] [um] [tsk]``, emotion tokens, and a vocabulary resize to add them.
**None of that is needed and some of it is wrong.** The model already ships:

* a closed set of **16 non-verbal tokens** at ids 156944–156959, including
  ``<|nv_breath|>``, ``<|nv_stammer|>`` and ``<|nv_laugh|>``;
* an ``<|env>…<env|>`` wrapper whose label is **open-class free text** — exactly the
  per-clip source token D-007 asks for, with no tokenizer change;
* ``<|speaker>`` and ``<|style>`` wrappers, and a documented prompt layout.

So there is nothing to add and nothing to resize. That matters beyond tidiness: the
contract warns that compiling data with one tokenizer and training with another fails
*silently* — "training runs, loss descends, and the model learns the wrong symbol for
every audio frame" — which is why ``TOKENIZER_FINGERPRINT`` must be asserted when data
is compiled and again when training starts.

Two corrections to the runbook's tag list, both from ``voices.md``:

* the trained emotion labels are ``ANGER``, not ``angry``, and there is **no
  ``neutral``**. Conditioning on a label the model never saw is silent misconditioning;
* styles are not only emotions — eight delivery styles were trained too.

And two things the shipped ``inference.py`` settles, which the other docs get wrong or
leave ambiguous:

* **``style`` is free text.** ``build_prompt`` validates nothing — it encodes
  ``enc(style.strip())``. The README: *"`style` is free text. Two forms appear in
  training: uppercase emotion labels (``ANGER``, ``FEAR``, … 17 in total) and
  descriptive phrases such as ``happiness, speaking with a strong texan accent``."*
  Only 14 of those 17 are documented anywhere, so refusing an unlisted style would block
  a form the model was trained on. ``is_known_style`` reports instead of refusing.
* **Where the prompt ends is genuinely ambiguous.** ``inference.py`` ends it with
  ``<|start_of_ai|>, <|start_of_speech|>``; ``token_contract.md`` §5 says the prompt stops
  *before* ``<|start_of_speech|>``, "which the model emits itself". The shipped code is
  what actually runs, so ``for_inference=True`` follows it; the default follows the
  contract, because §5's definition is what sets the loss boundary when training data is
  compiled. Getting this wrong at compile time trains the model to predict a token it is
  already given.

Note also that ``inference.py`` never emits an ``<|env>`` block: the wrapper exists in the
tokenizer but the shipped inference path does not exercise it. Stage A trains it, so that
is workable — but the base model's ``<|env>`` conditioning should not be assumed to work
before the pilot measures it.

One thing to keep straight: ``voices.md`` says the model "derives pauses and intonation
from punctuation, so place commas wherever a speaker would breathe". Punctuation is
therefore **load-bearing in training text and inference prompts**.
``geetha.text.normalise.norm_text`` strips it, and is for the leak hash and the scorers
**only** — never for building training text.
"""

from __future__ import annotations

__all__ = [
    "AUDIO_TOKEN_BASE_ID",
    "CODEBOOKS",
    "CODEBOOK_SIZE",
    "CONTROL_TOKENS",
    "CONDITIONING_TOKENS",
    "EMOTION_STYLES",
    "NONVERBAL_TOKENS",
    "SOURCE_LABELS",
    "STYLE_LABELS",
    "TOKENIZER_FINGERPRINT",
    "TOKENS_PER_SECOND",
    "VOCAB_SIZE",
    "build_tts_prompt",
    "estimate_audio_tokens",
    "is_known_style",
    "map_corpus_nonverbal",
]

TOKENIZER_FINGERPRINT = "b645cf6612315393eea21d29e94304bdbcbfee131a25de5eecf6bae6ee9f39c5"
VOCAB_SIZE = 156_960

# Audio band. The layout is discontinuous -- the control block sits immediately below
# it and the conditioning block immediately above -- so anything assuming "added tokens
# are contiguous above the base" is wrong.
AUDIO_TOKEN_BASE_ID = 128_266
CODEBOOKS = 7
CODEBOOK_SIZE = 4_096

# "82 tokens ~= 1 second of audio" (contract §3). Seven tokens per 12 Hz frame would be
# 84; consecutive duplicate frames are dropped at encode time, so 82 is the working
# figure and a token estimate is an upper bound, never an undercount.
TOKENS_PER_SECOND = 82

CONTROL_TOKENS = {
    "<|start_of_speech|>": 128_257,
    "<|end_of_speech|>": 128_258,
    "<|start_of_human|>": 128_259,
    "<|end_of_human|>": 128_260,
    "<|start_of_ai|>": 128_261,
    "<|end_of_ai|>": 128_262,
    "<|pad|>": 128_263,
}

CONDITIONING_TOKENS = {
    "<|speaker>": 156_938,
    "<speaker|>": 156_939,
    "<|style>": 156_940,
    "<style|>": 156_941,
    "<|env>": 156_942,
    "<env|>": 156_943,
}

NONVERBAL_TOKENS = {
    "<|nv_breath|>": 156_944,
    "<|nv_stammer|>": 156_945,
    "<|nv_throat|>": 156_946,
    "<|nv_laugh|>": 156_947,
    "<|nv_swallow|>": 156_948,
    "<|nv_sniff|>": 156_949,
    "<|nv_sigh|>": 156_950,
    "<|nv_cough|>": 156_951,
    "<|nv_hum|>": 156_952,
    "<|nv_gasp|>": 156_953,
    "<|nv_wheeze|>": 156_954,
    "<|nv_sneeze|>": 156_955,
    "<|nv_snort|>": 156_956,
    "<|nv_yawn|>": 156_957,
    "<|nv_groan|>": 156_958,
    "<|nv_burp|>": 156_959,
}

# The emotion labels the model was trained on, in the casing the README and the CLI
# both use (`--style ANGER`). The README says there are 17 in total; only these 6 plus
# the 8 delivery styles below are documented anywhere, so the set is a hint, not a
# whitelist.
EMOTION_STYLES = ("HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST")

# Delivery styles, trained alongside the emotions. Any voice combines with any style.
DELIVERY_STYLES = (
    "advertisements",
    "AIR style news",
    "TV style news",
    "single person narration audiobook",
    "children's stories",
    "educational lecture",
    "news",
    "Customer Care",
)

STYLE_LABELS = DELIVERY_STYLES + EMOTION_STYLES

# Our own <|env> values (D-007). Open class, so these are a project convention rather
# than anything the tokenizer constrains -- but they must stay a fixed, small set or
# the model cannot learn to separate them.
SOURCE_LABELS = ("studio", "enhanced", "phone_restored")

# Corpus source forms -> the model's own non-verbal tokens. Keys are the canonical
# names produced by geetha.text.annotated.canonical_tag and the IndicVoices
# unsanitized_verbatim marks, with the brackets already stripped.
#
# Deliberately absent:
#   * fillers ("uhh", "umm") -- speech, not events, and the closed set has no filler
#     token. They become Tamil filler text instead.
#   * environment noise ("noise", "bird_squawk", "horn", "unintelligible", ...) --
#     removed from the text, not voiced.
_CORPUS_NONVERBALS = {
    "breathing": "<|nv_breath|>",
    "breath": "<|nv_breath|>",
    "stammers": "<|nv_stammer|>",
    "stammer": "<|nv_stammer|>",
    "stutters": "<|nv_stammer|>",
    "throat_clearing": "<|nv_throat|>",
    "laughter": "<|nv_laugh|>",
    "laugh": "<|nv_laugh|>",
    "swallowing": "<|nv_swallow|>",
    "sniff": "<|nv_sniff|>",
    "sigh": "<|nv_sigh|>",
    "coughing": "<|nv_cough|>",
    "cough": "<|nv_cough|>",
    "hum": "<|nv_hum|>",
    "gasps": "<|nv_gasp|>",
    "gasp": "<|nv_gasp|>",
    "yawn": "<|nv_yawn|>",
    "groan": "<|nv_groan|>",
    "sneeze": "<|nv_sneeze|>",
    "snort": "<|nv_snort|>",
    "wheeze": "<|nv_wheeze|>",
    "burp": "<|nv_burp|>",
}


def map_corpus_nonverbal(name: str) -> str | None:
    """The model's non-verbal token for a corpus annotation, or ``None``.

    ``None`` means "this is not a non-verbal event the model can voice" — a filler, or
    environment noise. Callers must handle it explicitly rather than let an unmapped
    annotation fall through into text.
    """
    return _CORPUS_NONVERBALS.get(name.strip().lower().replace(" ", "_"))


def is_known_style(style: str) -> bool:
    """Whether ``style`` is one of the documented trained labels.

    Reports rather than refuses: style is free text and the README says 17 emotion
    labels exist where only 6 are documented. Use this to flag that a prompt has gone
    off the known set, not to block it.
    """
    return style.strip().upper() in {s.upper() for s in STYLE_LABELS}


def estimate_audio_tokens(seconds: float) -> int:
    """Upper bound on SNAC tokens for ``seconds`` of audio."""
    return int(seconds * TOKENS_PER_SECOND)


def build_tts_prompt(
    *,
    speaker: str,
    style: str,
    text: str,
    source: str | None = None,
    for_inference: bool = False,
) -> str:
    """Build a conditioned TTS prompt in the documented layout.

    Metadata order is always speaker → style → env, each block followed by a newline.
    An empty value emits nothing at all, not an empty wrapper.

    ``style`` is **not** validated: it is free text in the model (see the module
    docstring). Use ``is_known_style`` if you want to know whether it is on the
    documented trained set.

    ``for_inference=True`` appends ``<|start_of_speech|>``, matching the shipped
    ``inference.py``. The default stops at ``<|start_of_ai|>``, matching
    ``token_contract.md`` §5's definition of where a training prompt ends.
    """
    if source is not None and source not in SOURCE_LABELS:
        raise ValueError(
            f"source {source!r} is not one of {SOURCE_LABELS!r}; <|env> is open class "
            "but the project keeps it to a fixed set so the model can separate them"
        )

    blocks = []
    if speaker:
        blocks.append(f"<|speaker>{speaker}<speaker|>\n")
    if style:
        blocks.append(f"<|style>{style}<style|>\n")
    if source:
        blocks.append(f"<|env>{source}<env|>\n")

    prompt = (
        "<|start_of_human|><|begin_of_text|>"
        + "".join(blocks)
        + text
        + "<|eot_id|><|end_of_human|><|start_of_ai|>"
    )
    return prompt + "<|start_of_speech|>" if for_inference else prompt
