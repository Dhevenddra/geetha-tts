r"""B4a — did Indic-Speak train on Rasa's *test* split? (Kaggle, free GPU, ~1.5 h)

Runs the three-arm design documented in ``geetha/eval/contamination.py``. Read that
module first: the verdict rule was fixed there before any audio existed, and this script
only applies it.

    TRAIN   50 Rasa Tamil train clips  — positive control, near-certainly trained on
    TEST    50 Rasa Tamil test clips   — the arm under suspicion, matched to TRAIN on
                                         gender, style group and duration bin
    UNSEEN  50 FLEURS Tamil sentences  — a loose anchor for text from another corpus
                                         (CER only; its speakers are not Anitha/Arun)

Inputs come from the Kaggle dataset built by
``scripts/make_contamination_arms.py``: ``manifest.json`` and the real recordings.

Outputs to ``/kaggle/working``:

    c1_results.json     every per-clip number, the arm statistics and the verdict
    c1_report.md        the same thing in prose, for the eval report and STATUS
    generated/*.wav     our renderings, kept so the operator can listen

Judges, and their limits, stated here rather than buried:

* **CER judge** — ``openai/whisper-large-v3`` at ``language="ta"``. Tamil CER from
  Whisper is not strong in absolute terms. That is tolerable because the verdict rests
  on the *difference* between two arms scored by the same judge on the same day, not on
  the absolute level. The absolute numbers are not a baseline and must not be reported
  as one; B6 does baselines properly.
* **Speaker similarity** — ``microsoft/wavlm-base-plus-sv``. This is *not* the
  Seed-TTS-eval WavLM-large the release will pin (R15); it is the convenient one. Used
  here only to confirm the voice identity is what we asked for, never as a published
  number.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

IN_DIR = Path("/kaggle/input/geetha-c1-contamination")  # replaced at runtime by find_dataset
OUT_DIR = Path("/kaggle/working")
GEN_DIR = OUT_DIR / "generated"

MODEL_ID = "bodhan-ai/indic-speak"
ASR_ID = "openai/whisper-large-v3"
SV_ID = "microsoft/wavlm-base-plus-sv"

# voices.md says the voice cards were measured "at the production config
# (ckpt-8944, t 0.6 - top_p 0.9 - rp 1.2)". Matching it keeps us on the operating point
# the model was characterised at, so a low CER cannot be blamed on lucky sampling.
#
# But the README's signature is `tts(text, speaker, style, temperature, top_p, top_k,
# max_new_tokens, seed, stock)` -- there is **no repetition_penalty**. Passing the "rp"
# voices.md quotes would raise TypeError, and a blanket fallback would then quietly drop
# temperature and top_p too and run at library defaults, while this file went on
# claiming the production config. So: pass only what the signature documents, and if
# even that is refused, say so loudly in the log rather than degrade in silence.
# max_new_tokens was 2520, straight from the README's example. That is wrong for this
# job: every clip here is 3-12 s, and contract.py measured the model at 82 audio tokens
# per second, so 12 s is under 1,000 tokens. A clip that never emits a stop token was
# burning 2.5x more compute than it could possibly need -- and on a T4 that is the
# difference between a run that finishes and one that hits the wall clock.
MAX_SECONDS_PER_CLIP = 13.0
TOKENS_PER_SECOND = 82
GEN_KWARGS = dict(
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    max_new_tokens=int(MAX_SECONDS_PER_CLIP * TOKENS_PER_SECOND),
)
SEED = 20261027

SAMPLE_RATE = 24_000
F0_FRAMES = 100          # contours are resampled to this length before correlating
F0_MIN, F0_MAX = 60.0, 400.0


# --------------------------------------------------------------------------- distances


def f0_contour(wav: np.ndarray, sr: int) -> np.ndarray:
    """Voiced log-F0, resampled to a fixed length and z-normalised.

    Fixed length and z-normalisation strip absolute pitch and absolute duration, leaving
    the *shape* of the intonation. Duration is not thrown away -- it is measured
    separately in :func:`prosodic_distance`, which keeps the two signals legible instead
    of blending them into one opaque number.
    """
    import librosa

    f0 = librosa.yin(wav, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=1024)
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.isfinite(f0) & (f0 > F0_MIN) & (f0 < F0_MAX)
    if voiced.sum() < 8:
        return np.zeros(F0_FRAMES)
    logf0 = np.log(f0[voiced])
    resampled = np.interp(
        np.linspace(0.0, 1.0, F0_FRAMES),
        np.linspace(0.0, 1.0, len(logf0)),
        logf0,
    )
    spread = resampled.std()
    if spread < 1e-6:
        return np.zeros(F0_FRAMES)
    return (resampled - resampled.mean()) / spread


def prosodic_distance(
    contour_a: np.ndarray, duration_a: float, contour_b: np.ndarray, duration_b: float
) -> float:
    """Distance in [0, 2]: half intonation shape, half timing.

    Shape is ``1 - pearson(contour_a, contour_b)``. Timing is ``|log(dur_a / dur_b)|``,
    clipped at 1 -- a rendering twice or half the length of the real take is already as
    wrong as the measure needs to say.

    Timing carries real weight on purpose. A model reproducing a memorised take matches
    not only the words but *that take's* pauses and tempo, and duration is the cheapest,
    least arguable place that shows up.
    """
    if not np.any(contour_a) or not np.any(contour_b):
        shape = 1.0
    else:
        correlation = float(np.corrcoef(contour_a, contour_b)[0, 1])
        shape = 1.0 - (0.0 if math.isnan(correlation) else correlation)
    if duration_a <= 0 or duration_b <= 0:
        timing = 1.0
    else:
        timing = min(1.0, abs(math.log(duration_a / duration_b)))
    return 0.5 * shape + 0.5 * timing


# --------------------------------------------------------------------------- pipeline


def load_manifest() -> list[dict]:
    manifest = json.loads((IN_DIR / "manifest.json").read_text(encoding="utf-8"))
    print(f"manifest: {len(manifest['clips'])} clips, seed {manifest['seed']}")
    return manifest["clips"]


def fleurs_sentences(n: int) -> list[str]:
    """Tamil text from another corpus, for the UNSEEN CER anchor."""
    from datasets import load_dataset

    data = load_dataset("google/fleurs", "ta_in", split="test", streaming=True)
    out: list[str] = []
    for row in data:
        text = (row.get("raw_transcription") or row.get("transcription") or "").strip()
        # Match the Rasa arms' sentence length so CER is not compared across very
        # different utterance sizes.
        if 40 <= len(text) <= 220:
            out.append(text)
        if len(out) >= n:
            break
    print(f"FLEURS: {len(out)} sentences")
    return out


# A private Kaggle dataset holding one file: a read-only, repo-scoped Hugging Face
# token. Measured 27 Oct 2026: a kernel pushed through the API does **not** keep a
# secret attached in the UI -- version 4 died in 1.7 s with
# "Kaggle secret unavailable (ConnectionError)". Since every remaining Kaggle job
# (B6-B10) needs the same gated model, a path that survives an API push is not a
# convenience, it is the difference between jobs I can launch and jobs that need the
# operator present.
#
# The token in this dataset is deliberately *not* the write token in
# `.secrets\geetha.env`. It is fine-grained, read-only, and scoped to gated repos the
# account can already read, so the worst case if the dataset leaked is read access to
# models that are published anyway.
TOKEN_DATASET = Path("/kaggle/input/geetha-secrets/hf_read_token.txt")


def hf_token() -> tuple[str | None, str]:
    """The gated-model token, and where it came from.

    Three sources, in order of preference, because each fails in a different way:

    1. the private token dataset -- survives an API push, so this is the one that makes
       unattended runs possible;
    2. Kaggle Secrets -- works only when a human attached it through the UI and nobody
       has pushed since;
    3. the environment -- for running this script off Kaggle.

    The value is never printed or returned to a log. The *source* is, because when a run
    behaves oddly the first question is always which credential it used.
    """
    if TOKEN_DATASET.exists():
        token = TOKEN_DATASET.read_text(encoding="utf-8").strip()
        if token:
            return token, "private token dataset"

    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN"), "Kaggle Secrets"
    except Exception as error:
        print(f"Kaggle secret unavailable ({type(error).__name__})", flush=True)

    return os.environ.get("HF_TOKEN"), "environment"


def require_token() -> str:
    """Resolve the gated-model token and fail in seconds if it is missing.

    This runs as the very first thing in ``main``, before FLEURS is streamed or any
    model is fetched. A kernel pushed through the API does not carry its secret
    attachment in ``kernel-metadata.json``, so whether an attached secret survives a
    re-push is the single most likely thing to be wrong with this run -- and the worst
    possible place to discover it is forty minutes in. Failing fast turns that question
    into a one-minute answer we can read off the log.

    The token's value is never printed. Only where it came from, and its length.
    """
    token, source = hf_token()
    if not token:
        raise SystemExit(
            "NO HF TOKEN from any source. Either attach the geetha-secrets dataset to "
            "this kernel, or attach HF_TOKEN in the Kaggle UI (Add-ons -> Secrets) and "
            "re-run from the UI without pushing afterwards. The model is gated."
        )
    # Put it in the environment rather than passing it to one call by hand. Hugging
    # Face's libraries all read HF_TOKEN themselves, so this covers load_dataset and
    # from_pretrained as well as snapshot_download. Without it the FLEURS download goes
    # out anonymous ("You are sending unauthenticated requests to the HF Hub"), which
    # is harmless for a public dataset but takes the tighter rate limit -- and Kaggle
    # runs from shared IPs, where that is a real way to lose an hour to a 429.
    os.environ["HF_TOKEN"] = token
    print(f"HF token resolved from {source}, length {len(token)}.", flush=True)
    return token


def ensure_dependencies() -> None:
    """Install what the model's own inference.py needs, before anything imports it.

    Its docstring says: transformers>=5, torch, snac, soundfile. Kaggle ships torch and
    soundfile; ``snac`` is absent and transformers may be too old. transformers is only
    upgraded when it is actually below 5 -- replacing a working install costs minutes of
    a metered quota and can break the Whisper pipeline used later in this same run.
    """
    import subprocess

    def pip(*args: str) -> None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)

    try:
        import snac  # noqa: F401
        print("snac already present", flush=True)
    except ImportError:
        print("installing snac", flush=True)
        pip("snac")

    import transformers

    major = int(transformers.__version__.split(".")[0])
    print(f"transformers {transformers.__version__}", flush=True)
    if major < 5:
        print("  upgrading to transformers>=5 (the model's tokenizer needs it)", flush=True)
        pip("-U", "transformers>=5")


def build_tts(token: str):
    import torch
    from huggingface_hub import snapshot_download

    local = snapshot_download(MODEL_ID, token=token)
    sys.path.insert(0, local)
    from inference import TTS  # type: ignore

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, _ = torch.cuda.get_device_capability(0)
        # Turing (sm_75: the T4) has no native bfloat16. The weights are bf16, so on a
        # T4 torch either upcasts to fp32 -- 3.78 B params is about 15 GB against the
        # card's 16 GB -- or takes a slow emulation path. Either way it is several times
        # slower than the same work in fp16, which Turing does support. Worth knowing
        # from the log rather than inferring it from a run that will not end.
        print(f"GPU {name}, compute capability {major}.x, bf16 native: {major >= 8}",
              flush=True)

    torch.manual_seed(SEED)
    tts = TTS(local)

    # inference.py hardcodes dtype=torch.bfloat16. Turing has no native bf16, so on a
    # T4 that runs through a slow path -- and the T4 is what Kaggle gives us. fp16 is
    # natively supported there and uses the tensor cores. Converting after construction
    # leaves the model's own loading logic untouched, which matters because the Vocos
    # decoder and the SNAC quantizer are loaded in the same call and are happier in
    # float32 (the decode path already calls .float() on the way in).
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        print("LM converted to fp16 — bf16 is not native on this card", flush=True)
    return tts


_GENERATION_CONFIG_USED: dict = {}


def generate(tts, text: str, speaker: str, style: str, seed: int) -> "np.ndarray":
    """One rendering at the production operating point, or a loud complaint.

    The first call records which keyword arguments the installed ``inference.py``
    actually accepted, and that record is written into the results file. A reader can
    then see the sampling settings every number here was produced under, instead of
    trusting this module's comments about them.
    """
    import numpy as _np

    attempts = [dict(GEN_KWARGS, seed=seed), dict(GEN_KWARGS), {}]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            wav = tts(text, speaker=speaker, style=style, **kwargs)
        except TypeError as error:
            last_error = error
            continue
        if not _GENERATION_CONFIG_USED:
            _GENERATION_CONFIG_USED.update(kwargs)
            if not kwargs:
                print(
                    "!! WARNING: inference.py refused every documented sampling "
                    f"argument ({last_error}). Running at library defaults -- the "
                    "generation config reported with these results says so.",
                    flush=True,
                )
            else:
                print(f"generation config in use: {kwargs}", flush=True)
        return _np.asarray(wav, dtype=_np.float32).squeeze()
    raise RuntimeError(f"could not call the model at all: {last_error}")


def synthesise(tts, rows: list[dict]) -> dict[str, np.ndarray]:
    import soundfile as sf
    import torch

    import time

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    produced: dict[str, np.ndarray] = {}
    started = time.time()
    for index, row in enumerate(rows, 1):
        torch.manual_seed(SEED + index)
        clip_started = time.time()
        wav = generate(tts, row["text"], row["speaker"], row["style"], SEED + index)
        produced[row["key"]] = wav
        # Written as it is produced, not at the end. A run that hits the wall clock
        # should leave behind everything it had finished rather than nothing -- hours of
        # a 30 h weekly quota are not something to lose to a timeout.
        sf.write(GEN_DIR / f"{row['key']}.wav", wav, SAMPLE_RATE)

        # The first few clips are the only honest estimate of the total. Printing them
        # individually means a slow run announces itself in minutes, not hours.
        elapsed = time.time() - clip_started
        if index <= 5 or index % 10 == 0:
            rate = (time.time() - started) / index
            remaining = rate * (len(rows) - index) / 60
            print(f"  {index}/{len(rows)}  {elapsed:.1f}s this clip, "
                  f"{rate:.1f}s/clip, ~{remaining:.0f} min left", flush=True)
    return produced


def transcribe(wavs: dict[str, np.ndarray]) -> dict[str, str]:
    import torch
    from transformers import pipeline

    asr = pipeline(
        "automatic-speech-recognition",
        model=ASR_ID,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    out: dict[str, str] = {}
    keys = list(wavs)
    for start in range(0, len(keys), 8):
        batch = keys[start : start + 8]
        results = asr(
            [{"raw": wavs[k], "sampling_rate": SAMPLE_RATE} for k in batch],
            generate_kwargs={"language": "ta", "task": "transcribe"},
        )
        for key, result in zip(batch, results):
            out[key] = result["text"].strip()
        print(f"  transcribed {min(start + 8, len(keys))}/{len(keys)}", flush=True)
    return out


def speaker_embeddings(wavs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector

    extractor = AutoFeatureExtractor.from_pretrained(SV_ID)
    model = WavLMForXVector.from_pretrained(SV_ID)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    import librosa

    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key, wav in wavs.items():
            resampled = librosa.resample(wav.astype(np.float64), orig_sr=SAMPLE_RATE, target_sr=16_000)
            features = extractor([resampled], sampling_rate=16_000, return_tensors="pt")
            if torch.cuda.is_available():
                features = {k: v.cuda() for k, v in features.items()}
            embedding = model(**features).embeddings.squeeze().cpu().numpy()
            out[key] = embedding / (np.linalg.norm(embedding) + 1e-9)
    return out


# Kaggle does not mount a dataset at /kaggle/input/<slug>. Measured 27 Oct 2026, the
# layout is /kaggle/input/datasets/<owner>/<slug>/ -- which cost two runs and two trips
# through the operator's UI before a directory listing was added and showed it in one
# line. Walking a few levels and matching on contents is immune to the next such
# change; a hardcoded path is not.
_MAX_MOUNT_DEPTH = 4


def _directories(root: Path, depth: int = 0):
    """Directories under ``root``, a few levels deep. Audio folders are not descended."""
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def show_inputs() -> list[Path]:
    """List what is actually mounted, before anything depends on it.

    Two runs died on a hardcoded ``/kaggle/input/geetha-code`` that was not there, and
    from the log alone there was no way to tell whether the dataset had failed to
    upload, failed to attach, or mounted under a different name. One listing settles
    all three, and costs nothing.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        print("!! /kaggle/input does not exist at all", flush=True)
        return []
    mounted = sorted(_directories(root))
    print(f"/kaggle/input holds {len(mounted)} director(ies):", flush=True)
    for entry in mounted:
        children = sorted(p.name for p in entry.iterdir())[:6]
        print(f"  {entry.relative_to(root)}/  -> {children}", flush=True)
    return mounted


def find_dataset(*markers: str) -> Path:
    """Locate a mounted dataset by a file or folder it is known to contain.

    Addressing mounts by the name Kaggle happened to give them has now failed twice.
    Addressing them by their *contents* cannot fail the same way: whatever the mount is
    called, the dataset holding ``manifest.json`` is the manifest dataset.
    """
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(
        f"no mounted dataset contains {list(markers)}. Attach it in the notebook "
        "editor: right sidebar -> Input -> + Add Input -> Datasets."
    )


def import_geetha() -> None:
    """Make ``geetha.*`` importable from the uploaded code dataset.

    Kaggle flattens a dataset upload, so the mount holds the *contents* of the package
    (``eval/``, ``scoring/``, ``text/``) rather than a folder named ``geetha``. Copying
    it under a correctly named parent is cheaper than rewriting every import.
    """
    import shutil

    source = find_dataset("scoring", "text", "eval")
    print(f"code package found at {source}", flush=True)
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


# Two rival accounts of what the model's `style` field actually takes, and a control.
#
# voices.md lists descriptive phrases ("advertisements", "educational lecture"). But Rasa
# Tamil's own `style` column holds sixteen terse uppercase labels — CONV, WIKI, BOOK,
# NEWS, ALEXA, UMANG, BB, DIGI, INDIC, PROPER NOUN and the six emotions — and the
# README says the style field takes "uppercase emotion labels … 17 in total". Several of
# the descriptive phrases look like renamings of Rasa labels (BOOK -> "single person
# narration audiobook", ALEXA -> "Customer Care"), so which spelling the model was
# actually conditioned on is genuinely open.
#
# It matters for naturalness, not just tidiness: if `CONV` is a real trained style, the
# base model already has a conversational register we can ask for by name.
#
# The test is cheap and decisive. `style` is free text and is never validated, so an
# unknown label is not an error — it just lands somewhere averaged. Render the same
# sentence, same speaker, same seed, under each candidate and against a nonsense
# control. A label the model never saw should sit no further from the control than the
# control's own run-to-run wobble.
STYLE_PROBE_LABELS = [
    "ZZQX",                              # nonsense control
    "CONV",                              # Rasa's label
    "advertisements",                    # voices.md's phrasing
    "BOOK",
    "single person narration audiobook",
    "ANGER",
    "anger",                             # casing: README says upper, voices.md lower
]
STYLE_PROBE_TEXT = (
    "நாளைக்கு காலையில கிளம்பணும், அதுக்கு முன்னாடி எல்லாத்தையும் தயார் பண்ணிடு."
)


def style_probe(tts) -> dict:
    """Does the model distinguish these style spellings at all?"""
    import soundfile as sf
    import torch

    probe_dir = OUT_DIR / "style_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, np.ndarray] = {}
    for label in STYLE_PROBE_LABELS:
        torch.manual_seed(SEED)
        wav = generate(tts, STYLE_PROBE_TEXT, "Anitha", label, SEED)
        rendered[label] = wav
        sf.write(probe_dir / f"{label.replace(' ', '_')}.wav", wav, SAMPLE_RATE)

    control = rendered[STYLE_PROBE_LABELS[0]]
    control_contour = f0_contour(control.astype(np.float64), SAMPLE_RATE)
    control_duration = len(control) / SAMPLE_RATE
    distances = {}
    for label, wav in rendered.items():
        distances[label] = prosodic_distance(
            f0_contour(wav.astype(np.float64), SAMPLE_RATE),
            len(wav) / SAMPLE_RATE,
            control_contour,
            control_duration,
        )
    print("style probe — distance from the nonsense control:")
    for label, distance in sorted(distances.items(), key=lambda kv: -kv[1]):
        print(f"  {distance:.3f}  {label}")
    return {"text": STYLE_PROBE_TEXT, "distance_from_control": distances}


def main() -> int:
    global IN_DIR

    # Diagnostics before gates. require_token() aborts in under two seconds when the
    # secret is missing, and if it ran first it would hide whether the datasets
    # mounted - which is the other thing that has gone wrong, and the one that cannot
    # be checked from this machine.
    show_inputs()
    token = require_token()
    ensure_dependencies()
    import_geetha()
    IN_DIR = find_dataset("manifest.json", "audio")
    print(f"clips found at {IN_DIR}", flush=True)
    from geetha.eval.contamination import (
        memorisation_margin,
        naturalness_source,
        verdict_from_gaps,
    )
    from geetha.scoring.bootstrap import two_sample_ci
    from geetha.scoring.metrics import cer
    from geetha.text.normalise import norm_text

    import librosa
    import soundfile as sf

    clips = load_manifest()
    for row in clips:
        row["key"] = f"{row['arm']}__{row['filename']}"

    rasa_rows = [r for r in clips if r["arm"] in ("TEST", "TRAIN")]
    unseen_rows = [
        {
            "arm": "UNSEEN",
            "filename": f"FLEURS_{i:03d}",
            "key": f"UNSEEN__FLEURS_{i:03d}",
            "text": text,
            "speaker": "Anitha" if i % 2 == 0 else "Arun",
            "gender": "Female" if i % 2 == 0 else "Male",
            "style": "WIKI",
            "style_group": "READ",
        }
        for i, text in enumerate(fleurs_sentences(50))
    ]

    tts = build_tts(token)
    probe = style_probe(tts)
    print(f"generating {len(rasa_rows) + len(unseen_rows)} clips", flush=True)
    generated = synthesise(tts, rasa_rows + unseen_rows)

    del tts
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    print("transcribing", flush=True)
    hypotheses = transcribe(generated)

    # ---- CER per clip, our own scorer and our own normalisation
    by_key = {r["key"]: r for r in rasa_rows + unseen_rows}
    cer_by_arm: dict[str, list[float]] = {"TEST": [], "TRAIN": [], "UNSEEN": []}
    per_clip: list[dict] = []
    for key, hypothesis in hypotheses.items():
        row = by_key[key]
        counts = cer(norm_text(hypothesis), norm_text(row["text"]))
        rate = 100.0 * counts.errors / max(1, counts.reference_length)
        cer_by_arm[row["arm"]].append(rate)
        per_clip.append({"key": key, "arm": row["arm"], "cer": rate, "hypothesis": hypothesis})

    # ---- prosody: matched vs a null pool of other real takes by the same voice
    print("measuring prosody", flush=True)
    real_cache: dict[str, tuple[np.ndarray, float]] = {}

    def real_clip(filename: str) -> tuple[np.ndarray, float]:
        if filename not in real_cache:
            wav, sr = sf.read(IN_DIR / "audio" / f"{filename}.wav", dtype="float64")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            real_cache[filename] = (f0_contour(wav, sr), len(wav) / sr)
        return real_cache[filename]

    null_by_cell: dict[tuple[str, str], list[str]] = {}
    for row in clips:
        if row["arm"] == "NULL":
            null_by_cell.setdefault((row["gender"], row["style_group"]), []).append(row["filename"])

    matched_by_arm: dict[str, list[float]] = {"TEST": [], "TRAIN": []}
    mismatched_by_arm: dict[str, list[float]] = {"TEST": [], "TRAIN": []}
    for row in rasa_rows:
        synth = generated[row["key"]]
        synth_contour = f0_contour(synth.astype(np.float64), SAMPLE_RATE)
        synth_duration = len(synth) / SAMPLE_RATE
        true_contour, true_duration = real_clip(row["filename"])
        matched_by_arm[row["arm"]].append(
            prosodic_distance(synth_contour, synth_duration, true_contour, true_duration)
        )
        for other in null_by_cell.get((row["gender"], row["style_group"]), []):
            other_contour, other_duration = real_clip(other)
            mismatched_by_arm[row["arm"]].append(
                prosodic_distance(synth_contour, synth_duration, other_contour, other_duration)
            )
        row["synth_duration"] = synth_duration

    # ---- speaker similarity, matched pairs only (a sanity check, never a headline)
    print("speaker similarity", flush=True)
    wanted = {r["key"]: generated[r["key"]] for r in rasa_rows}
    for row in rasa_rows:
        wav, sr = sf.read(IN_DIR / "audio" / f"{row['filename']}.wav", dtype="float64")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        wanted[f"REAL__{row['filename']}"] = wav.astype(np.float32)
    embeddings = speaker_embeddings(wanted)
    similarity_by_arm: dict[str, list[float]] = {"TEST": [], "TRAIN": []}
    for row in rasa_rows:
        similarity_by_arm[row["arm"]].append(
            float(embeddings[row["key"]] @ embeddings[f"REAL__{row['filename']}"])
        )

    # ---- the pre-registered verdict
    delta = {
        arm: memorisation_margin(matched_by_arm[arm], mismatched_by_arm[arm])
        for arm in ("TEST", "TRAIN")
    }
    cer_gap = two_sample_ci(cer_by_arm["TEST"], cer_by_arm["TRAIN"])

    # The margin is a ratio of medians, so it has no per-clip value to resample. Its
    # interval comes from resampling the matched distances of each arm against that
    # arm's own null median -- the null pool is shared and large, so treating its median
    # as fixed is the conservative simplification, not a convenient one.
    null_median = {arm: float(np.median(mismatched_by_arm[arm])) for arm in ("TEST", "TRAIN")}
    margin_samples = {
        arm: [(null_median[arm] - value) / null_median[arm] for value in matched_by_arm[arm]]
        for arm in ("TEST", "TRAIN")
    }
    delta_gap = two_sample_ci(margin_samples["TRAIN"], margin_samples["TEST"])

    verdict = verdict_from_gaps(
        cer_gap.delta, (cer_gap.low, cer_gap.high), delta_gap.delta, (delta_gap.low, delta_gap.high)
    )
    source = naturalness_source(verdict)

    results = {
        "verdict": verdict,
        "naturalness_ta_source": source,
        "cer_gap": {"delta": cer_gap.delta, "ci": [cer_gap.low, cer_gap.high], "p": cer_gap.p_value},
        "delta_gap": {
            "delta": delta_gap.delta,
            "ci": [delta_gap.low, delta_gap.high],
            "p": delta_gap.p_value,
        },
        "arms": {
            arm: {
                "n": len(cer_by_arm[arm]),
                "median_cer": float(np.median(cer_by_arm[arm])),
                "memorisation_margin": delta.get(arm),
                "median_matched_distance": float(np.median(matched_by_arm[arm]))
                if arm in matched_by_arm
                else None,
                "median_null_distance": null_median.get(arm),
                "median_speaker_similarity": float(np.median(similarity_by_arm[arm]))
                if arm in similarity_by_arm
                else None,
            }
            for arm in ("TEST", "TRAIN", "UNSEEN")
        },
        "style_probe": probe,
        "judges": {"cer": ASR_ID, "speaker_similarity": SV_ID},
        "generation_requested": GEN_KWARGS,
        "generation_actually_used": dict(_GENERATION_CONFIG_USED),
        "seed": SEED,
        "per_clip": per_clip,
    }
    (OUT_DIR / "c1_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# B4a - is Rasa's test split inside Indic-Speak?",
        "",
        f"**Verdict: {verdict}.** Naturalness-Ta is built from `{source}`.",
        "",
        "| arm | n | median CER | memorisation margin | median speaker sim |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in ("TRAIN", "TEST", "UNSEEN"):
        stats = results["arms"][arm]
        margin = "-" if stats["memorisation_margin"] is None else f"{stats['memorisation_margin']:.3f}"
        similarity = "-" if stats["median_speaker_similarity"] is None else f"{stats['median_speaker_similarity']:.3f}"
        lines.append(f"| {arm} | {stats['n']} | {stats['median_cer']:.2f} | {margin} | {similarity} |")
    lines += [
        "",
        f"- CER gap (TEST - TRAIN): **{cer_gap.delta:+.2f}** points, 95% CI "
        f"[{cer_gap.low:+.2f}, {cer_gap.high:+.2f}], p = {cer_gap.p_value:.3f}",
        f"- Margin gap (TRAIN - TEST): **{delta_gap.delta:+.3f}**, 95% CI "
        f"[{delta_gap.low:+.3f}, {delta_gap.high:+.3f}], p = {delta_gap.p_value:.3f}",
        "",
        "Rule, fixed in `geetha/eval/contamination.py` before any audio was generated: ",
        "CLEAN needs both gaps material (>= 2.0 CER points, >= 0.05 margin) and both CIs ",
        "clear of zero. Both CIs spanning zero is CONTAMINATED. Anything else is AMBIGUOUS, ",
        "and AMBIGUOUS is handled exactly as CONTAMINATED.",
        "",
        f"CER judge {ASR_ID}; speaker similarity {SV_ID} (not the Seed-TTS-eval WavLM the ",
        "release will pin). Absolute CER here is a within-run contrast, not a baseline.",
    ]
    (OUT_DIR / "c1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
