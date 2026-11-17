r"""C2 — what can the base model already do, and where is our headroom? (Kaggle, free)

Two questions, both cheap, both decision-relevant before a rupee is spent on training.

Question 1: is ``CONV`` a real trained style?
---------------------------------------------
``style`` is free text that the model never validates, so an unknown label does not
error -- it lands somewhere averaged. Two rival accounts of what it takes:

* ``voices.md`` lists descriptive phrases ("advertisements", "educational lecture");
* Rasa Tamil's own ``style`` column holds 16 terse uppercase labels (``CONV``, ``WIKI``,
  ``BOOK``, ``NEWS``, ``ALEXA``, the six emotions …), and the README says the field
  takes "uppercase emotion labels … 17 in total".

It matters for naturalness, not tidiness. If ``CONV`` is genuinely trained, the base
model already has a conversational register we can ask for **by name**, before any
LoRA -- which would change Stage A/B design and give Directability-Ta a real baseline.

The C1 probe failed to settle this because it compared the nonsense control against
*itself*, which is self-distance, not evidence. The fix is a **second nonsense label**:
``ZZQX`` vs ``QQWE`` establishes how far apart two labels the model has certainly never
seen land. Any real label must beat that gap to count as doing anything at all.

Seeds are varied too, because two renderings of the same label differ anyway. Without
that noise floor, "the label changed the output" is unfalsifiable.

Question 2: are the non-verbal tokens dead in the base model?
--------------------------------------------------------------
The model card's Limitations say plainly: *"No sound effects. No laughter, sighs, or
breaths."* Yet the tokenizer ships 16 non-verbal tokens at ids 156944-156959,
``<|nv_breath|>`` among them (D-017).

If those tokens are inert, that is the single clearest capability gap we can open: the
base model cannot breathe, and Geetha -- trained on ~74 h of wideband spontaneous Tamil
carrying real breath and filler marks (D-018) -- can. A capability gap is a better
claim than a score, because nobody can argue it away with a different metric.

It also tests R16's worry from the other side. ``tie_word_embeddings`` is true, so if
the base model emits these tokens *never*, training them is the whole job, and PEFT
trainable-tokens has to actually move the LM head.

Outputs to ``/kaggle/working``: ``c2_results.json``, ``c2_report.md``, and every
rendering, so the operator can listen rather than take a number on trust.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "bodhan-ai/indic-speak"
SAMPLE_RATE = 24_000
SEED = 20261027

MAX_SECONDS_PER_CLIP = 13.0
TOKENS_PER_SECOND = 82
GEN_KWARGS = dict(
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    max_new_tokens=int(MAX_SECONDS_PER_CLIP * TOKENS_PER_SECOND),
)

# Three seeds per (label, sentence). Two would give a difference; three gives a spread,
# which is what the comparison actually needs.
SEEDS = (20261027, 20261028, 20261029)

SENTENCES = [
    "நாளைக்கு காலையில கிளம்பணும், அதுக்கு முன்னாடி எல்லாத்தையும் தயார் பண்ணிடு.",
    "இந்த புத்தகத்தை நான் ரொம்ப நாளா தேடிட்டு இருந்தேன், கடைசியா கிடைச்சிடுச்சு.",
    "அவரு சொன்ன விஷயம் முழுக்க முழுக்க உண்மை, அதுல எந்த சந்தேகமும் இல்ல.",
]

# Two nonsense controls. The distance between THEM is the noise floor every real label
# has to beat. Chosen to be pronounceable-looking but meaningless in any language the
# model saw, and the same length, so length alone cannot explain a difference.
NONSENSE = ("ZZQX", "QQWE")

RASA_LABELS = ("CONV", "WIKI", "BOOK", "NEWS", "ANGER", "HAPPY")
VOICES_MD_LABELS = ("advertisements", "educational lecture", "Customer Care")
CASING = ("anger", "Anger")
STYLE_LABELS = NONSENSE + RASA_LABELS + VOICES_MD_LABELS + CASING + ("",)

# The non-verbal tokens, written inline the way training data would carry them.
NV_TOKENS = ("<|nv_breath|>", "<|nv_laugh|>", "<|nv_stammer|>")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


_MAX_MOUNT_DEPTH = 4


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def show_inputs() -> None:
    root = Path("/kaggle/input")
    print(f"/kaggle/input: {[str(p.relative_to(root)) for p in _directories(root)]}", flush=True)


def find_dataset(*markers: str) -> Path:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"no mounted dataset contains {list(markers)}")


def require_token() -> str:
    token = ""
    source = "none"
    secrets = Path("/kaggle/input")
    for mount in sorted(_directories(secrets)):
        candidate = mount / "hf_read_token.txt"
        if candidate.exists():
            token = candidate.read_text(encoding="utf-8").strip()
            source = "private token dataset"
            break
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("HF_TOKEN")
            source = "Kaggle Secrets"
        except Exception as error:
            print(f"Kaggle secret unavailable ({type(error).__name__})", flush=True)
    if not token:
        token = os.environ.get("HF_TOKEN", "")
        source = "environment"
    if not token:
        raise SystemExit("NO HF TOKEN from any source — attach the geetha-secrets dataset.")
    os.environ["HF_TOKEN"] = token
    print(f"HF token from {source}, length {len(token)}", flush=True)
    return token


def ensure_dependencies() -> None:
    import subprocess

    try:
        import snac  # noqa: F401
    except ImportError:
        print("installing snac", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snac"], check=False)


def build_tts(token: str):
    import torch
    from huggingface_hub import snapshot_download

    local = snapshot_download(MODEL_ID, token=token)
    sys.path.insert(0, local)
    from inference import TTS  # type: ignore

    tts = TTS(local)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        print("LM in fp16 (no native bf16 on this card)", flush=True)
    return tts


# ----------------------------------------------------------------- acoustic measures


def f0_stats(wav: np.ndarray) -> tuple[float, float]:
    """Median pitch and pitch variability, in semitones — the register and the life."""
    import librosa

    f0 = librosa.yin(wav.astype(np.float64), fmin=60.0, fmax=400.0, sr=SAMPLE_RATE,
                     frame_length=2048)
    voiced = f0[(f0 > 65) & (f0 < 395) & np.isfinite(f0)]
    if voiced.size < 8:
        return 0.0, 0.0
    semitones = 12 * np.log2(voiced / np.median(voiced))
    return float(np.median(voiced)), float(np.std(semitones))


def describe(wav: np.ndarray, text: str) -> dict:
    duration = len(wav) / SAMPLE_RATE
    pitch, variability = f0_stats(wav)
    # Speaking rate in characters per second — voices.md plans scripts this way
    # (library median ~12 chars/sec), so it is directly comparable to the voice cards.
    return {
        "duration": round(duration, 3),
        "chars_per_second": round(len(text) / duration, 2) if duration else 0.0,
        "pitch_hz": round(pitch, 1),
        "pitch_variability_semitones": round(variability, 3),
    }


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mel-spectrogram distance, length-normalised. Bounded enough to average."""
    import librosa

    def mel(wav: np.ndarray) -> np.ndarray:
        spec = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=SAMPLE_RATE,
                                              n_mels=64)
        spec = librosa.power_to_db(spec, ref=np.max)
        # Resample along time so clips of different length can be compared at all.
        return np.stack([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, spec.shape[1]), row)
                         for row in spec])

    ma, mb = mel(a), mel(b)
    return float(np.sqrt(np.mean((ma - mb) ** 2)))


def main() -> int:
    import soundfile as sf
    import torch

    show_inputs()
    token = require_token()
    ensure_dependencies()
    import_geetha()

    from geetha.text.contract import NONVERBAL_TOKENS, VOCAB_SIZE

    tts = build_tts(token)

    # --- tokenizer sanity: are the non-verbal ids where D-017 says they are?
    tok = tts.tok
    nv_ids = {name: tok.convert_tokens_to_ids(name) for name in NV_TOKENS}
    print(f"non-verbal token ids: {nv_ids}", flush=True)

    # D-017 recorded these ids by reading the model's files. Assert rather than trust:
    # the contract warns that a tokenizer mismatch fails *silently* -- "training runs,
    # loss descends, and the model learns the wrong symbol for every audio frame" --
    # and a probe reporting on the wrong ids would be worse than no probe.
    for name, found in nv_ids.items():
        expected = NONVERBAL_TOKENS.get(name) if isinstance(NONVERBAL_TOKENS, dict) else None
        if expected is not None and found != expected:
            raise SystemExit(f"token id mismatch for {name}: contract says {expected}, "
                             f"this tokenizer says {found}. Stop — D-017 is stale.")
    print(f"vocab {len(tok)} (contract expects {VOCAB_SIZE})", flush=True)

    audio_dir = OUT_DIR / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[tuple[str, int, int], np.ndarray] = {}
    measures: list[dict] = []

    def render(text: str, style: str, seed: int, tag: str) -> np.ndarray:
        torch.manual_seed(seed)
        wav = tts(text, speaker="Anitha", style=style, seed=seed, **GEN_KWARGS)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        sf.write(audio_dir / f"{tag}.wav", wav, SAMPLE_RATE)
        return wav

    # ---------------------------------------------------- Q1: does style do anything?
    total = len(STYLE_LABELS) * len(SENTENCES) * len(SEEDS)
    print(f"Q1: {total} renderings across {len(STYLE_LABELS)} style labels", flush=True)
    done = 0
    for li, label in enumerate(STYLE_LABELS):
        for si, sentence in enumerate(SENTENCES):
            for ki, seed in enumerate(SEEDS):
                safe = (label or "EMPTY").replace(" ", "_").replace("|", "")
                wav = render(sentence, label, seed, f"style_{safe}_s{si}_k{ki}")
                rendered[(label, si, ki)] = wav
                measures.append({"label": label, "sentence": si, "seed": seed,
                                 **describe(wav, sentence)})
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{total}", flush=True)

    # Noise floor: same label, different seed.
    within: list[float] = []
    for label in STYLE_LABELS:
        for si in range(len(SENTENCES)):
            for a in range(len(SEEDS)):
                for b in range(a + 1, len(SEEDS)):
                    within.append(spectral_distance(rendered[(label, si, a)],
                                                    rendered[(label, si, b)]))
    noise_floor = float(np.median(within))

    # The control gap: two labels the model has certainly never seen.
    control_gap = float(np.median([
        spectral_distance(rendered[(NONSENSE[0], si, ki)], rendered[(NONSENSE[1], si, ki)])
        for si in range(len(SENTENCES)) for ki in range(len(SEEDS))
    ]))

    # Every real label, measured against the nonsense controls.
    label_gap: dict[str, float] = {}
    for label in STYLE_LABELS:
        if label in NONSENSE:
            continue
        distances = [
            spectral_distance(rendered[(label, si, ki)], rendered[(control, si, ki)])
            for control in NONSENSE
            for si in range(len(SENTENCES))
            for ki in range(len(SEEDS))
        ]
        label_gap[label] = float(np.median(distances))

    # ------------------------------------------- Q2: are the non-verbal tokens inert?
    print("Q2: non-verbal tokens", flush=True)
    nv_results = []
    for si, sentence in enumerate(SENTENCES):
        plain = rendered[("", si, 0)]
        for name in NV_TOKENS:
            # Inline, mid-sentence, where a speaker would actually breathe.
            words = sentence.split(" ")
            middle = len(words) // 2
            marked = " ".join(words[:middle] + [name] + words[middle:])
            wav = render(marked, "", SEEDS[0], f"nv_{name.strip('<>|')}_s{si}")
            nv_results.append({
                "token": name,
                "token_id": nv_ids[name],
                "sentence": si,
                "duration_plain": round(len(plain) / SAMPLE_RATE, 3),
                "duration_marked": round(len(wav) / SAMPLE_RATE, 3),
                "distance_from_plain": round(spectral_distance(plain, wav), 4),
            })

    nv_distance = float(np.median([r["distance_from_plain"] for r in nv_results]))

    # ------------------------------------------------------------------- the verdicts
    # A label "does something" only if it moves the output further than two labels the
    # model has never seen move it from each other.
    style_is_real = {label: gap > control_gap for label, gap in label_gap.items()}
    rasa_wins = sum(style_is_real.get(label, False) for label in RASA_LABELS)
    voices_wins = sum(style_is_real.get(label, False) for label in VOICES_MD_LABELS)

    results = {
        "noise_floor_same_label_different_seed": round(noise_floor, 4),
        "control_gap_nonsense_vs_nonsense": round(control_gap, 4),
        "label_gap_vs_nonsense": {k: round(v, 4) for k, v in label_gap.items()},
        "label_beats_control": style_is_real,
        "rasa_labels_beating_control": f"{rasa_wins}/{len(RASA_LABELS)}",
        "voices_md_labels_beating_control": f"{voices_wins}/{len(VOICES_MD_LABELS)}",
        "nonverbal": {
            "token_ids": nv_ids,
            "median_distance_from_plain": round(nv_distance, 4),
            "inert_if_close_to": round(noise_floor, 4),
            "appears_inert": nv_distance <= noise_floor,
            "per_clip": nv_results,
        },
        "measures": measures,
        "generation": GEN_KWARGS,
        "seeds": list(SEEDS),
    }
    (OUT_DIR / "c2_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                             encoding="utf-8")

    lines = [
        "# C2 — what the base model can already do",
        "",
        f"Noise floor (same label, different seed): **{noise_floor:.3f}**",
        f"Control gap (two nonsense labels): **{control_gap:.3f}**",
        "",
        "A style label only counts as *doing something* if it moves the output further",
        "than two labels the model has certainly never seen move it from each other.",
        "",
        "| style label | distance from nonsense | beats control? |",
        "| --- | --- | --- |",
    ]
    for label, gap in sorted(label_gap.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{label or '(empty)'}` | {gap:.3f} | "
                     f"{'yes' if style_is_real[label] else 'no'} |")
    lines += [
        "",
        f"Rasa's own labels beating control: **{rasa_wins}/{len(RASA_LABELS)}**; "
        f"voices.md phrases: **{voices_wins}/{len(VOICES_MD_LABELS)}**.",
        "",
        "## Non-verbal tokens",
        "",
        f"Median distance from the unmarked sentence: **{nv_distance:.3f}** "
        f"against a noise floor of {noise_floor:.3f}.",
        "",
        f"**{'INERT' if nv_distance <= noise_floor else 'HAS AN EFFECT'}** — the model "
        "card's Limitations say \"No sound effects. No laughter, sighs, or breaths.\"",
        "If inert, that is our clearest capability gap: the base model cannot breathe,",
        "and Stage A has ~74 h of marked spontaneous Tamil to teach it (D-018).",
    ]
    (OUT_DIR / "c2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
