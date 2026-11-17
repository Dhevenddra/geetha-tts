r"""Pre-M2: does the trained pilot produce Tamil at all? (free Kaggle GPU)

Why this run exists
-------------------
Gate M2 asks for ₹1,000 to buy a 25-30 h Stage A run. All four of its opening
conditions currently have **no evidence whatsoever**, for one root reason:

    **Nobody has ever rendered audio from a trained Geetha.**

m1-pilot-11 finished on 6 Nov with a Stage B loss of 3.838 and 25 checkpoints on the
Hub. Its adapter has been inspected tensor-by-tensor (D-039, D-049, D-050) and never
once asked to speak. Buying 25 paid hours before hearing 10 seconds of output is the
definition of spending to find out what a free run would have told us.

So this run asks the only question that matters before the gate: **is the output
Tamil-shaped speech, or is it noise?** It is not trying to pass M2. A 20-minute-per-stage
pilot will not approach the base model, and the honest expectation is a *worse* CER than
base. The failure this guards against is categorical, not marginal: a token-contract
mismatch, a mis-compiled codebook, a loader bug - any of which produce a healthy loss
curve and undecodable audio, which is exactly how the gemma3 SNAC base mismatch failed
(token_contract.md).

Design: paired, not absolute
----------------------------
41 prompts from the frozen B7 set, chosen for being **single-chunk** so no chunking,
rejoin, forced-split or truncation artefact can enter - B7 itself truncated 37 prompts
and re-split others. Each carries B7's own CER **and its own hypothesis** for the same
prompt id, so the base model's exact error counts are recomputed here and fed to a
`paired_bootstrap` -- a delta with a CI and a p-value, and no baseline re-render. Mean
B7 CER on this subset: **38.80**; the full 262-prompt set was 41.81.

The same judge (`openai/whisper-large-v3`), the same decoding (temperature 0.6, top_p
0.9, top_k 50) and **the same per-prompt seed** B7 used (`SEED + b7_index*100`, the
index recovered from B7's clip order) are reused. So the comparison is a paired delta on
identical prompts under identical conditions, which is the only form in which a CER from
a 3B TTS model means anything: absolute CER here is dominated by the judge's own error on
synthetic Tamil and by the codec, which C5 measured at **+14.6 CER** all on its own.

The adapter is attached **unmerged**, per D-049: merging changes what the model emits
(0 of 192 greedy tokens identical), so the merged artefact is not the model we trained.

Second measurement: does the breath tag do anything?
----------------------------------------------------
D-050 is open - every breath in the corpus belongs to an IndicVoices speaker, and the two
voices we ship have none. This run cannot settle transfer on a 20-minute pilot, but it can
build and exercise the instrument.

Six sentence pairs, rendered with and without `<|nv_breath|>` at a fixed seed, scored with
C2's own mel-spectrogram distance so the numbers are comparable to C2's reading of the base
model (`<|nv_breath|>` moved the audio **15.166** against a noise floor of **16.773** -
i.e. inert).

The noise floor is measured **in this run, on this model**, by re-rendering the same text
at a different seed. Reusing C2's 16.773 would compare our signal to another model's noise.
If the tag's effect does not clear our own floor, the tag does nothing - the same shape of
answer C2 gave for the base model.

Ten clips are written out for the operator to listen to, which is M2's third condition.

Spends no money: free Kaggle GPU, no gate, no pod.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "bodhan-ai/indic-speak"
# Pinned, like C5 and the D-049 kernel. The model repo ships `inference.py` and the Vocos
# decoder as **importable Python**, so an unpinned `main` means third-party code can
# change under us between runs -- which is both a supply-chain exposure and a silent
# confound: B7's baseline was rendered from this repo on 27 Oct, and a paired comparison
# against those numbers is only valid if the code and weights are identical. (They are:
# the repo was last updated 7 Sep 2026.)
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
ASR_ID = "openai/whisper-large-v3"
JUDGE_RATE = 16_000  # Whisper accepts nothing else; our audio is 24 kHz
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTER_SUBDIR = "final/m1-pilot-11/adapter_final"
SAMPLE_RATE = 24_000

MAX_SECONDS_PER_CHUNK = 22.0
TOKENS_PER_SECOND = 82
GEN_KWARGS = dict(
    temperature=0.6,
    top_p=0.9,
    top_k=50,
    max_new_tokens=int(MAX_SECONDS_PER_CHUNK * TOKENS_PER_SECOND),
)
SEED = 20261027  # B7's seed, so per-prompt seeds reproduce exactly
N_LISTEN = 10

# 41 single-chunk, untruncated, unresplit B7 prompts, each carrying B7's clip index (so
# its seed reproduces), its CER, and **its hypothesis** -- which lets the base model's
# exact error counts be recomputed here and fed to a paired bootstrap, with no need to
# re-render the baseline. Mean B7 CER on this subset: 38.80 (full 262-prompt set: 41.81).
# Built from runs\b7_baseline_rev2\b7_results.json into dhevenddra/geetha-m2pre-subset.
SUBSET_JSON = "m2pre_subset.json"

# Pairs for the breath A/B. Real Stage A training text containing exactly one
# <|nv_breath|>; the control is the same sentence with the tag removed. Training text,
# deliberately, not eval_vault: the question is whether the token does anything at all,
# and spending a frozen eval item on a throwaway pilot buys nothing.
BREATH_PAIRS = [
    "இல்ல <|nv_breath|> ஆட்டாங்கல்லோ அந்த மாரி இருந்தா",
    "கோவில்லேர்ந்து <|nv_breath|> அப்படியே வந்துட்டேன்",
    "அது <|nv_breath|> ஒரு பெரிய விஷயம் இல்ல",
    "நான் <|nv_breath|> அங்கே போகலைன்னு சொன்னேன்",
    "அப்பரம் <|nv_breath|> ரெண்டு நாள் ஆட்டம்",
    "அப்படி தான் <|nv_breath|> அவர் சொன்னார்",
]
BREATH_TAG = "<|nv_breath|>"
BREATH_SEED = 777
BREATH_FLOOR_SEED = 778  # same text, different roll -> this model's own noise floor

_MAX_MOUNT_DEPTH = 4


def log(message: str) -> None:
    print(message, flush=True)


def fix_environment() -> None:
    """Kaggle ships torchao 0.10.0, which makes PEFT unable to build any LoRA.

    `is_torchao_available()` **raises** on a present-but-too-old torchao instead of
    returning False, and PEFT calls it for every target module. Killed the first D-049
    kernel outright. Must happen before torch/transformers/peft are imported, because
    `find_spec` consults `sys.modules` first.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        capture_output=True, text=True,
    )
    log(f"  pip uninstall torchao -> {result.returncode}")


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    """Address inputs by their contents, never by a vendor's path.

    Datasets mount at /kaggle/input/datasets/<owner>/<slug>/, not /kaggle/input/<slug>/.
    Three runs went into diagnosing an "attachment problem" that never existed.
    """
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"could not find a mount containing {markers}")


def show_inputs() -> None:
    log("inputs:")
    for mount in sorted(_directories(Path("/kaggle/input"))):
        log(f"  {mount}")


def require_token() -> str:
    path = find_dataset("hf_read_token.txt") / "hf_read_token.txt"
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("hf_read_token.txt is empty")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def ensure_dependencies() -> None:
    import subprocess

    # Pinned rather than `-U`: an unpinned upgrade changes what runs between this run
    # and the next, which for a measurement we intend to compare against is a confound
    # as much as a supply-chain question.
    # These are the versions `-U` resolved on 6 Nov 2026, pinned so the source matches
    # the run that produced the numbers. librosa 1.0.0 is a major bump since C2 ran in
    # October, which is one more reason the breath floor is measured in *this* run
    # rather than compared against C2's stored 16.773.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "snac==1.2.1", "soundfile==0.14.0", "librosa==1.0.0",
                    "jiwer==4.0.0"], check=True)
    log("installed snac 1.2.1, soundfile 0.14.0, librosa 1.0.0, jiwer 4.0.0 (pinned)")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def _load_subset() -> list[dict]:
    path = find_dataset(SUBSET_JSON) / SUBSET_JSON
    rows = json.loads(path.read_text(encoding="utf-8"))["clips"]
    rows.sort(key=lambda row: row["b7_index"])
    log(f"  subset: {len(rows)} prompts, B7 mean CER "
        f"{sum(r['b7_cer'] for r in rows) / len(rows):.2f}")
    return rows


def build_tts(token: str, adapter: Path):
    """The model repo's own TTS wrapper, with our LoRA attached but NOT merged.

    Unmerged is the deliberate choice: D-049 measured that merging changes what the
    model emits (0 of 192 greedy tokens identical), because `lm_head` is tied to
    `embed_tokens` and sits outside `target_modules`, so the output head only sees the
    delta once the merge writes it in. The unmerged model is the one training optimised.
    """
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    local = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)

    # Everything gated or private is on disk by now: the adapter was fetched in main()
    # and the base model on the line above. `inference.py` below is third-party code
    # from the model repo, executed in this process, so the read token comes out of the
    # environment before it is imported. It can still read `geetha-ckpt-private`, and
    # nothing after this point needs it from the environment -- the ASR download passes
    # its token explicitly.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)

    sys.path.insert(0, local)
    from inference import TTS  # type: ignore

    tts = TTS(local)
    before = sum(p.numel() for p in tts.lm.parameters())
    tts.lm = PeftModel.from_pretrained(tts.lm, str(adapter))
    tts.lm.eval()
    after = sum(p.numel() for p in tts.lm.parameters())
    log(f"  adapter attached (unmerged): {before:,} -> {after:,} parameters "
        f"(+{after - before:,})")
    if after <= before:
        raise SystemExit("attaching the adapter added no parameters; it did not load")

    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        log("  LM in fp16 (no native bf16 on this card)")
    return tts


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """C2's metric, copied verbatim so the numbers are comparable to C2's reading."""
    import librosa

    def mel(wav: np.ndarray) -> np.ndarray:
        spec = librosa.feature.melspectrogram(y=wav.astype(np.float32), sr=SAMPLE_RATE,
                                              n_mels=64)
        spec = librosa.power_to_db(spec, ref=np.max)
        return np.stack([np.interp(np.linspace(0, 1, 100),
                                   np.linspace(0, 1, spec.shape[1]), row)
                         for row in spec])

    ma, mb = mel(a), mel(b)
    return float(np.sqrt(np.mean((ma - mb) ** 2)))


def main() -> int:
    log("=== 0. environment ===")
    fix_environment()

    import soundfile as sf
    import torch

    show_inputs()
    token = require_token()
    ensure_dependencies()
    import_geetha()

    import peft
    import transformers
    log(f"  torch {torch.__version__}, transformers {transformers.__version__}, "
        f"peft {peft.__version__}")
    log(f"  gpu: {torch.cuda.get_device_name(0)}")

    from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap
    from geetha.scoring.metrics import cer

    log("")
    log("=== 1. inputs ===")
    subset = _load_subset()
    prompts_dir = find_dataset("prompts.json")
    clips = {row["id"]: row
             for row in json.loads((prompts_dir / "prompts.json")
                                   .read_text(encoding="utf-8"))["clips"]}
    missing = [row["id"] for row in subset if row["id"] not in clips]
    if missing:
        raise SystemExit(f"prompt ids missing from prompts.json: {missing[:5]}")

    from huggingface_hub import snapshot_download
    adapter_root = snapshot_download(
        CKPT_REPO, allow_patterns=[f"{ADAPTER_SUBDIR}/*"], token=token
    )
    adapter = Path(adapter_root) / ADAPTER_SUBDIR
    log(f"  adapter: {adapter}")

    log("")
    log("=== 2. model ===")
    tts = build_tts(token, adapter)

    audio_dir = OUT_DIR / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)

    def render(text: str, speaker: str, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        wav = tts(text, speaker=speaker, style="", seed=seed, **GEN_KWARGS)
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    log("")
    log("=== 3. rendering 41 prompts ===")
    started = time.time()
    rendered: dict[str, np.ndarray] = {}
    for position, row in enumerate(subset, 1):
        clip_id = row["id"]
        # B7's own per-prompt seed, so the pair differs only by the adapter.
        seed = SEED + row["b7_index"] * 100
        wav = render(clips[clip_id]["text"], row["speaker"], seed)
        rendered[clip_id] = wav
        sf.write(audio_dir / f"{clip_id}.wav", wav, SAMPLE_RATE)
        if position % 5 == 0 or position == len(subset):
            elapsed = time.time() - started
            log(f"  {position}/{len(subset)}  {elapsed / position:.1f} s/prompt  "
                f"eta {(len(subset) - position) * elapsed / position / 60:.1f} min")

    log("")
    log("=== 4. judging ===")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(ASR_ID, token=token)
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_ID, token=token, dtype=torch.float16
    ).to("cuda").eval()

    def transcribe(wav: np.ndarray) -> str:
        # Whisper refuses anything but 16 kHz and our audio is 24 kHz. C5 died here on
        # its own v2 and left the comment; version 1 of this kernel died here too,
        # having rendered all 41 clips first. torchaudio handles the 3:2 ratio properly,
        # and both arms are judged identically -- the base arm's hypothesis came from
        # B7, which resampled the same way -- so this cannot bias the delta.
        import torchaudio

        resampled = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), SAMPLE_RATE, JUDGE_RATE
        ).numpy()
        features = processor(resampled, sampling_rate=JUDGE_RATE,
                            return_tensors="pt").input_features.to("cuda").half()
        with torch.inference_mode():
            ids = asr.generate(features, language="ta", task="transcribe",
                               max_new_tokens=220)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    per_clip = []
    ours_counts: list = []
    base_counts: list = []
    for row in subset:
        clip_id = row["id"]
        reference = clips[clip_id]["text"]
        hypothesis = transcribe(rendered[clip_id])
        # cer() takes (hypothesis, reference) in that order and normalises both itself:
        # swapping them would divide by the hypothesis length instead of the reference's.
        mine = cer(hypothesis, reference)
        # The base model's counts, recomputed from B7's own stored hypothesis for this
        # same prompt. Exact, and it costs no GPU time: the baseline is not re-rendered.
        theirs = cer(row["b7_hypothesis"], reference)
        ours_counts.append(mine)
        base_counts.append(theirs)
        per_clip.append({
            "id": clip_id, "category": row["category"], "speaker": row["speaker"],
            "geetha_cer": round(100.0 * (mine.rate or 0.0), 2),
            "base_cer_recomputed": round(100.0 * (theirs.rate or 0.0), 2),
            "b7_cer_as_published": row["b7_cer"],
            "seconds": round(len(rendered[clip_id]) / SAMPLE_RATE, 2),
            "hypothesis": hypothesis,
        })

    ours = bootstrap_ci(ours_counts)
    base = bootstrap_ci(base_counts)
    paired = paired_bootstrap(ours_counts, base_counts)
    worse = sum(1 for row in per_clip
                if row["geetha_cer"] > row["base_cer_recomputed"])

    log("")
    log("=== 5. result ===")
    log(f"  Geetha (m1-pilot-11, unmerged): CER {100 * ours.point:.2f} "
        f"[{100 * ours.low:.2f}, {100 * ours.high:.2f}]")
    log(f"  Indic-Speak base, same prompts, same judge: CER {100 * base.point:.2f} "
        f"[{100 * base.low:.2f}, {100 * base.high:.2f}]")
    log(f"  paired delta: {100 * paired.delta:+.2f} CER points "
        f"[{100 * paired.low:+.2f}, {100 * paired.high:+.2f}], p = {paired.p_value:.4f}")
    log(f"  worse than base on {worse}/{len(per_clip)} prompts")
    log(f"  median clip length {np.median([r['seconds'] for r in per_clip]):.2f} s")
    # Version 1 crashed before judging but its 41 clips already answered the
    # categorical question: 39/41 emitted <|end_of_speech|> on their own, 35 distinct
    # durations, and duration correlated r = 0.658 with sentence length. Babble would
    # not track the text, so this is reported as a first-class number, not a footnote.
    pinned = sum(1 for row in per_clip if row["seconds"] >= 21.7)
    lengths = [len(clips[row["id"]]["text"]) for row in per_clip]
    correlation = float(np.corrcoef(lengths, [r["seconds"] for r in per_clip])[0, 1])
    log(f"  terminated before the token cap: {len(per_clip) - pinned}/{len(per_clip)}")
    log(f"  duration vs sentence length: r = {correlation:.3f} "
        f"(unconditioned babble would be ~0)")
    log("  (a worse CER here is expected: 20 minutes of training per stage. The")
    log("   question this run answers is whether the audio is Tamil at all.)")

    log("")
    log("=== 6. breath A/B ===")
    breath = []
    for index, marked in enumerate(BREATH_PAIRS):
        plain = marked.replace(BREATH_TAG, " ").replace("  ", " ").strip()
        with_tag = render(marked, "anitha", BREATH_SEED + index)
        without = render(plain, "anitha", BREATH_SEED + index)
        # This model's own noise floor: identical text, a different roll.
        reroll = render(plain, "anitha", BREATH_FLOOR_SEED + index)
        signal = spectral_distance(with_tag, without)
        floor = spectral_distance(without, reroll)
        breath.append({
            "index": index, "signal": round(signal, 4), "floor": round(floor, 4),
            "clears_floor": signal > floor,
            "seconds_with": round(len(with_tag) / SAMPLE_RATE, 2),
            "seconds_without": round(len(without) / SAMPLE_RATE, 2),
        })
        sf.write(audio_dir / f"breath_{index}_with.wav", with_tag, SAMPLE_RATE)
        sf.write(audio_dir / f"breath_{index}_without.wav", without, SAMPLE_RATE)
        log(f"  pair {index}: tag moved {signal:.3f}, this model's floor {floor:.3f} "
            f"-> {'clears' if signal > floor else 'INSIDE THE FLOOR'}")

    mean_signal = float(np.mean([b["signal"] for b in breath]))
    mean_floor = float(np.mean([b["floor"] for b in breath]))
    cleared = sum(1 for b in breath if b["clears_floor"])
    log(f"  mean: tag {mean_signal:.3f} vs floor {mean_floor:.3f}; "
        f"{cleared}/{len(breath)} pairs clear")
    log(f"  (C2 on the base model: 15.166 against a floor of 16.773 = inert)")

    report = {
        "adapter": f"{CKPT_REPO}/{ADAPTER_SUBDIR}",
        "merged": False,
        "judge": ASR_ID,
        "generation": GEN_KWARGS,
        "seed_rule": "SEED + b7_index*100, B7's own per-prompt seed",
        "n": len(per_clip),
        "geetha_cer": round(100 * ours.point, 2),
        "geetha_ci": [round(100 * ours.low, 2), round(100 * ours.high, 2)],
        "base_cer_same_prompts": round(100 * base.point, 2),
        "base_ci": [round(100 * base.low, 2), round(100 * base.high, 2)],
        "paired_delta": round(100 * paired.delta, 2),
        "paired_delta_ci": [round(100 * paired.low, 2), round(100 * paired.high, 2)],
        "paired_p_value": paired.p_value,
        "worse_than_base": worse,
        "terminated_before_cap": len(per_clip) - pinned,
        "duration_vs_length_r": round(correlation, 3),
        "per_clip": per_clip,
        "breath": breath,
        "breath_mean_signal": round(mean_signal, 4),
        "breath_mean_floor": round(mean_floor, 4),
        "breath_pairs_clearing_floor": cleared,
        "c2_base_reading": {"signal": 15.166, "floor": 16.773, "verdict": "inert"},
    }
    (OUT_DIR / "m2pre_pilot_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    listen = sorted(per_clip, key=lambda row: row["geetha_cer"])
    picks = listen[:N_LISTEN // 2] + listen[-(N_LISTEN - N_LISTEN // 2):]
    (OUT_DIR / "LISTEN.md").write_text(
        "# Ten clips to listen to\n\n"
        "Five best and five worst by CER. The question is not whether they are good -- "
        "20 minutes of training per stage -- but whether they are **Tamil**.\n\n"
        + "\n".join(f"- `clips/{row['id']}.wav` ({row['category']}, {row['speaker']}, "
                    f"CER {row['geetha_cer']} vs base {row['base_cer_recomputed']}): "
                    f"heard as “{row['hypothesis'][:90]}”"
                    for row in picks)
        + "\n\nAlso `clips/breath_*_with.wav` against `clips/breath_*_without.wav`: "
          "the same sentence with and without the breath tag.\n",
        encoding="utf-8",
    )
    log("")
    log(f"wrote {OUT_DIR / 'm2pre_pilot_eval.json'} and LISTEN.md; "
        f"{len(list(audio_dir.iterdir()))} wavs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
