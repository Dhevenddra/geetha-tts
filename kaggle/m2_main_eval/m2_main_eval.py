r"""M2 main run: the first numbers and the first audio from a full-length Geetha.

m2-main-1 trained Stage A for 9 h and Stage B for 3 h. Its held-out loss fell; nobody has
heard it and no eval number exists. This kernel is the reading (free Kaggle GPU, no gate).

What is different from ``m2pre_pilot_eval`` (D-051, and two things found on 11 Nov):

* **Geetha's own prompt layout.** speaker + optional style + ``<|env>studio<env|>``, each
  piece encoded with ``add_special_tokens=False``, exactly as ``geetha.train.corpus`` builds
  a training row. The pilot eval went through ``inference.TTS`` with no env and no style.
* **Lower-case speaker names.** Every training row says ``anitha`` / ``arun``; B7 and the
  pilot eval rendered ``Anitha`` / ``Arun``. A casing arm measures what that costs.
* **The loudness gate.** Every clip passes ``geetha.eval.validity.check_clip`` before a judge
  hears it; invalid clips are their own rate and never enter a CER pool.
* **A repetition-penalty arm.** The Orpheus README says ``repetition_penalty>=1.1`` "is
  required for stable generations". Neither the base model's ``inference.py`` nor any Geetha
  eval has ever set one, and ~5-10% of pilot clips were mute or ran away (D-052).
* **A positive control for the breath tag.** The same pairs under an IndicVoices speaker,
  who *did* breathe in training. Without it an inert tag on Anitha cannot be told apart
  from a tag that was never learned at all (D-050). Control audio is never written to disk:
  IndicVoices speakers are training signal only (D-007).

Arms on the 41 paired B7 prompts (B7's judge, decoding and per-prompt seeds):

    B_plain   Stage B adapter, no style            <- the headline pair against base
    A_plain   Stage A adapter, no style            <- is Stage B worth its 3 hours?
    B_rp11    Stage B, repetition_penalty 1.1      <- does the missing penalty matter?
    B_conv    Stage B, style CONV, 16 prompts      <- which default should the card give?
    B_Case    Stage B, "Anitha"/"Arun", 8 prompts  <- cost of the casing mismatch

The report is rewritten after every section, so a late failure still leaves the early arms.
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
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"  # pinned, as in every kernel
ASR_ID = "openai/whisper-large-v3"
JUDGE_RATE = 16_000
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTERS = {"stageB": "m2-main/stageB/final", "stageA": "m2-main/stageA/final"}
SAMPLE_RATE = 24_000

MAX_SECONDS_PER_CHUNK = 22.0
TOKENS_PER_SECOND = 82
GEN_KWARGS = dict(temperature=0.6, top_p=0.9, top_k=50,
                  max_new_tokens=int(MAX_SECONDS_PER_CHUNK * TOKENS_PER_SECOND))
SEED = 20261027  # B7's seed, so per-prompt seeds reproduce exactly
SUBSET_JSON = "m2pre_subset.json"

HEAD = "<|start_of_human|><|begin_of_text|>"
TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"

BREATH_TAG = "<|nv_breath|>"
# The first six are real Stage A training text (IndicVoices), as in the pilot eval. The
# last four are written for this run. None is an eval-vault item.
BREATH_PAIRS = [
    "இல்ல <|nv_breath|> ஆட்டாங்கல்லோ அந்த மாரி இருந்தா",
    "கோவில்லேர்ந்து <|nv_breath|> அப்படியே வந்துட்டேன்",
    "அது <|nv_breath|> ஒரு பெரிய விஷயம் இல்ல",
    "நான் <|nv_breath|> அங்கே போகலைன்னு சொன்னேன்",
    "அப்பரம் <|nv_breath|> ரெண்டு நாள் ஆட்டம்",
    "அப்படி தான் <|nv_breath|> அவர் சொன்னார்",
    "சரி <|nv_breath|> நாளைக்கு பாக்கலாம்",
    "எனக்கு தெரியல <|nv_breath|> நீயே சொல்லு",
    "அவங்க வந்தாங்க <|nv_breath|> ஆனா ஒண்ணும் சொல்லல",
    "இப்போ <|nv_breath|> என்ன பண்ணலாம்னு யோசிக்கிறேன்",
]
BREATH_SEED = 777
BREATH_FLOOR_SEED = 778
# The IndicVoices speaker with the most breath marks in Stage A (194 over 115 rows).
CONTROL = dict(speaker="ivr_f251dc9c", style="extempore", env="enhanced")
# (adapter, voice, number of pairs). Control rows write no audio.
BREATH_PLAN = [
    ("stageB", "anitha", 10), ("stageB", "arun", 5), ("stageB", "control", 5),
    ("stageA", "anitha", 6), ("stageA", "control", 4),
]

# Two neutral carriers from eval_vault\directability_ta_v1.jsonl (c01, c02; neither is one
# of the six D-033 flagged as not neutral). DISGUST is in training but not in the vault.
EMOTION_CARRIERS = {
    "c01": "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.",
    "c02": "அந்த புத்தகம் மேஜை மேல இருக்கு.",
}
EMOTION_STYLES = ["CONV", "HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST"]
EMOTION_SEED = 4242

_MAX_MOUNT_DEPTH = 4


def log(message: str) -> None:
    print(message, flush=True)


def fix_environment() -> None:
    """Kaggle's torchao 0.10.0 makes PEFT unable to build any LoRA; remove it first."""
    import subprocess

    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            capture_output=True, text=True)
    log(f"  pip uninstall torchao -> {result.returncode}")


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    """Address inputs by their contents, never by a vendor's path."""
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"could not find a mount containing {markers}")


def require_token() -> str:
    path = find_dataset("hf_read_token.txt") / "hf_read_token.txt"
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("hf_read_token.txt is empty")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def ensure_dependencies() -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "snac==1.2.1", "soundfile==0.14.0", "librosa==1.0.0",
                    "jiwer==4.0.0"], check=True)
    log("installed snac 1.2.1, soundfile 0.14.0, librosa 1.0.0, jiwer 4.0.0 (pinned)")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    if not (source / "eval" / "validity.py").exists():
        raise SystemExit("geetha-code has no eval/validity.py: re-push the code dataset")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def fetch_adapters(token: str) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    found = {}
    for name, subdir in ADAPTERS.items():
        root = Path(snapshot_download(CKPT_REPO, allow_patterns=[f"{subdir}/*"],
                                      token=token))
        folder = root / subdir
        if not (folder / "adapter_config.json").exists():
            nested = sorted(folder.rglob("adapter_config.json"))
            if not nested:
                raise SystemExit(f"no adapter_config.json under {subdir}")
            folder = nested[0].parent
        log(f"  {name}: {folder}  ({sorted(p.name for p in folder.iterdir())})")
        found[name] = folder
    return found


def build_tts(token: str, adapters: dict[str, Path]):
    """The base repo's TTS wrapper with both adapters attached, NOT merged (D-049)."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    local = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)
    # inference.py is third-party code run in this process: everything private is on
    # disk by now, so the token leaves the environment before it is imported.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)
    sys.path.insert(0, local)
    from inference import TTS  # type: ignore

    tts = TTS(local)
    before = sum(p.numel() for p in tts.lm.parameters())
    names = list(adapters)
    tts.lm = PeftModel.from_pretrained(tts.lm, str(adapters[names[0]]),
                                       adapter_name=names[0])
    for name in names[1:]:
        tts.lm.load_adapter(str(adapters[name]), adapter_name=name)
    tts.lm.eval()
    after = sum(p.numel() for p in tts.lm.parameters())
    log(f"  adapters attached (unmerged): {before:,} -> {after:,} parameters")
    if after <= before:
        raise SystemExit("attaching the adapters added no parameters")
    if torch.cuda.get_device_capability(0)[0] < 8:
        tts.lm = tts.lm.half()
        log("  LM in fp16 (no native bf16 on this card)")
    return tts


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """C2's metric, verbatim, so the numbers stay comparable with C2 and the pilot."""
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

    for mount in sorted(_directories(Path("/kaggle/input"))):
        log(f"  input: {mount}")
    token = require_token()
    ensure_dependencies()
    import_geetha()

    import peft
    import transformers
    log(f"  torch {torch.__version__}, transformers {transformers.__version__}, "
        f"peft {peft.__version__}; gpu: {torch.cuda.get_device_name(0)}")

    from geetha.eval.validity import check_clip, summarise
    from geetha.scoring.bootstrap import bootstrap_ci, paired_bootstrap
    from geetha.scoring.metrics import cer

    log("")
    log("=== 1. inputs ===")
    subset_path = find_dataset(SUBSET_JSON) / SUBSET_JSON
    subset = json.loads(subset_path.read_text(encoding="utf-8"))["clips"]
    subset.sort(key=lambda row: row["b7_index"])
    prompts_dir = find_dataset("prompts.json")
    clips = {row["id"]: row
             for row in json.loads((prompts_dir / "prompts.json")
                                   .read_text(encoding="utf-8"))["clips"]}
    missing = [row["id"] for row in subset if row["id"] not in clips]
    if missing:
        raise SystemExit(f"prompt ids missing from prompts.json: {missing[:5]}")
    log(f"  {len(subset)} paired prompts; B7 mean CER "
        f"{sum(r['b7_cer'] for r in subset) / len(subset):.2f}")
    adapters = fetch_adapters(token)

    log("")
    log("=== 2. model ===")
    tts = build_tts(token, adapters)
    from inference import build_prompt, ids_to_codes  # type: ignore

    tok = tts.tok
    start_speech = tok.convert_tokens_to_ids("<|start_of_speech|>")
    stop_ids = {tts._eos, tok.eos_token_id}

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def geetha_prompt(text: str, speaker: str, style: str = "",
                      env: str = "studio") -> list[int]:
        """One training row's prompt, piece by piece (geetha.train.corpus.Item)."""
        return (enc(HEAD) + enc(f"<|speaker>{speaker}<speaker|>\n")
                + (enc(f"<|style>{style}<style|>\n") if style else [])
                + (enc(f"<|env>{env}<env|>\n") if env else [])
                + enc(text) + enc(TAIL) + [start_speech])

    # D-051 says our layout with no style and no env IS the base layout. Prove it on
    # this tokenizer before spending two hours measuring anything through it.
    probe = clips[subset[0]["id"]]["text"]
    ours, theirs = geetha_prompt(probe, "Anitha", env=""), build_prompt(tok, probe, "Anitha")
    if ours != theirs:
        log(f"  ours  : {ours[:24]} ... {ours[-8:]}")
        log(f"  theirs: {theirs[:24]} ... {theirs[-8:]}")
        raise SystemExit("Geetha's prompt layout does not reduce to the base layout")
    log(f"  layout check passed ({len(ours)} tokens identical to the base build_prompt)")
    log(f"  'anitha' -> {enc('anitha')}   'Anitha' -> {enc('Anitha')}")

    def generate(prompt: list[int], seed: int, **overrides):
        """-> (wav, ended_itself, new_tokens). An undecodable output is an empty clip."""
        torch.manual_seed(seed)
        ids = torch.tensor([prompt], device=tts.device)
        kwargs = {**GEN_KWARGS, **overrides}
        with torch.no_grad():
            gen = tts.lm.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                eos_token_id=[tts._eos, tok.eos_token_id],
                pad_token_id=tok.eos_token_id, do_sample=True, **kwargs)
            new = gen[0].tolist()[len(prompt):]
            ended = bool(new) and new[-1] in stop_ids
            try:
                codes = ids_to_codes(new, tts._base, tts.device)
                wav = tts.vocos(tts.snac.quantizer.from_codes(codes).float())
                wav = wav[0, 0].clamp(-1, 1).float().cpu().numpy()
            except ValueError:
                wav = np.zeros(0, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32).reshape(-1), ended, len(new)

    def validity(wav: np.ndarray, ended: bool):
        return check_clip(wav, SAMPLE_RATE, ended_itself=ended)

    audio_dir = OUT_DIR / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "adapters": {k: f"{CKPT_REPO}/{v}" for k, v in ADAPTERS.items()},
        "merged": False, "judge": ASR_ID, "generation": GEN_KWARGS,
        "seed_rule": "SEED + b7_index*100, B7's own per-prompt seed",
        "layout": "speaker(lower-case) + optional style + <|env>studio<env|>, "
                  "add_special_tokens=False",
        "arms": {}, "breath": {}, "emotion": [],
    }

    def flush() -> None:
        (OUT_DIR / "m2_main_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ rendering
    arm_specs = [
        # name, adapter, rows, style, lower-case speaker, generation overrides
        ("B_plain", "stageB", subset, "", True, {}),
        ("A_plain", "stageA", subset, "", True, {}),
        ("B_rp11", "stageB", subset, "", True, {"repetition_penalty": 1.1}),
        ("B_conv", "stageB", subset[:16], "CONV", True, {}),
        ("B_Case", "stageB", subset[:8], "", False, {}),
    ]
    rendered: dict[str, dict[str, tuple]] = {}
    log("")
    log("=== 3. rendering the CER arms ===")
    for name, adapter, rows, style, lower, overrides in arm_specs:
        tts.lm.set_adapter(adapter)
        started = time.time()
        out = {}
        for position, row in enumerate(rows, 1):
            speaker = row["speaker"].lower() if lower else row["speaker"]
            prompt = geetha_prompt(clips[row["id"]]["text"], speaker, style)
            wav, ended, n_new = generate(prompt, SEED + row["b7_index"] * 100, **overrides)
            out[row["id"]] = (wav, validity(wav, ended), n_new)
            if name == "B_plain" and len(wav):
                sf.write(audio_dir / f"{row['id']}.wav", wav, SAMPLE_RATE)
            if position % 10 == 0 or position == len(rows):
                log(f"  {name} {position}/{len(rows)}  "
                    f"{(time.time() - started) / position:.1f} s/prompt")
        rendered[name] = out
        summary = summarise(check for _, check, _ in out.values())
        log(f"  {name}: {summary.line()}")
        report["arms"][name] = {
            "adapter": adapter, "style": style, "lower_case_speaker": lower,
            "overrides": overrides, "n": len(rows), "validity": summary.line(),
            "invalid_rate": round(100 * summary.invalid_rate, 2),
            "silent": summary.silent, "hit_cap": summary.hit_cap,
        }
        flush()

    # ------------------------------------------------------------------ breath A/B
    log("")
    log("=== 4. breath A/B, with an IndicVoices positive control ===")
    for adapter, voice, count in BREATH_PLAN:
        tts.lm.set_adapter(adapter)
        who = CONTROL if voice == "control" else dict(speaker=voice, style="CONV",
                                                      env="studio")
        pairs = []
        for index, marked in enumerate(BREATH_PAIRS[:count]):
            plain = marked.replace(BREATH_TAG, " ").replace("  ", " ").strip()
            with_tag, e1, _ = generate(geetha_prompt(marked, **who), BREATH_SEED + index)
            without, e2, _ = generate(geetha_prompt(plain, **who), BREATH_SEED + index)
            reroll, e3, _ = generate(geetha_prompt(plain, **who),
                                     BREATH_FLOOR_SEED + index)
            ok = all(validity(w, e).valid
                     for w, e in ((with_tag, e1), (without, e2), (reroll, e3)))
            entry = {"index": index, "valid": ok,
                     "seconds_with": round(len(with_tag) / SAMPLE_RATE, 2),
                     "seconds_without": round(len(without) / SAMPLE_RATE, 2),
                     "seconds_reroll": round(len(reroll) / SAMPLE_RATE, 2)}
            if ok:
                entry["signal"] = round(spectral_distance(with_tag, without), 4)
                entry["floor"] = round(spectral_distance(without, reroll), 4)
                entry["clears_floor"] = entry["signal"] > entry["floor"]
            pairs.append(entry)
            if voice != "control":  # D-007: IndicVoices audio is never written out
                for label, wav in (("with", with_tag), ("without", without)):
                    if len(wav):
                        sf.write(audio_dir / f"breath_{adapter}_{voice}_{index}_{label}.wav",
                                 wav, SAMPLE_RATE)
        good = [p for p in pairs if p["valid"]]
        result = {"pairs": pairs, "valid_pairs": len(good)}
        if good:
            result.update(
                mean_signal=round(float(np.mean([p["signal"] for p in good])), 4),
                mean_floor=round(float(np.mean([p["floor"] for p in good])), 4),
                clearing_floor=sum(1 for p in good if p["clears_floor"]),
                # A breath takes time: the tag should lengthen the clip more than a
                # re-roll does.
                mean_extra_seconds_tag=round(float(np.mean(
                    [p["seconds_with"] - p["seconds_without"] for p in good])), 3),
                mean_extra_seconds_reroll=round(float(np.mean(
                    [p["seconds_reroll"] - p["seconds_without"] for p in good])), 3))
            log(f"  {adapter}/{voice}: {result['clearing_floor']}/{len(good)} valid pairs "
                f"clear the floor; tag {result['mean_signal']:.3f} vs floor "
                f"{result['mean_floor']:.3f}; tag adds "
                f"{result['mean_extra_seconds_tag']:+.2f} s, re-roll "
                f"{result['mean_extra_seconds_reroll']:+.2f} s")
        else:
            log(f"  {adapter}/{voice}: no valid pair out of {len(pairs)}")
        report["breath"][f"{adapter}/{voice}"] = result
        flush()
    report["breath"]["references"] = {
        "c2_base": {"signal": 15.166, "floor": 16.773, "verdict": "inert"},
        "pilot": "3/6 pairs cleared their floor = no detectable effect"}

    # ------------------------------------------------------------------ emotion set
    log("")
    log("=== 5. emotion clips for the listen (Geetha Stage B, then the base model) ===")
    tts.lm.set_adapter("stageB")
    for carrier, text in EMOTION_CARRIERS.items():
        for voice in ("anitha", "arun"):
            for style in EMOTION_STYLES:
                wav, ended, _ = generate(geetha_prompt(text, voice, style), EMOTION_SEED)
                check = validity(wav, ended)
                name = f"emo_geetha_{carrier}_{voice}_{style}.wav"
                if len(wav):
                    sf.write(audio_dir / name, wav, SAMPLE_RATE)
                report["emotion"].append({
                    "system": "geetha_stageB", "file": name, "carrier": carrier,
                    "voice": voice, "style": style, "valid": check.valid,
                    "reason": check.reason, "seconds": round(check.seconds, 2)})
    with tts.lm.disable_adapter():
        for voice in ("Anitha", "Arun"):
            for style in EMOTION_STYLES:
                text = EMOTION_CARRIERS["c01"]
                prompt = build_prompt(tok, text, voice, "" if style == "CONV" else style)
                wav, ended, _ = generate(prompt, EMOTION_SEED)
                check = validity(wav, ended)
                name = f"emo_base_c01_{voice.lower()}_{style}.wav"
                if len(wav):
                    sf.write(audio_dir / name, wav, SAMPLE_RATE)
                report["emotion"].append({
                    "system": "base", "file": name, "carrier": "c01",
                    "voice": voice, "style": style, "valid": check.valid,
                    "reason": check.reason, "seconds": round(check.seconds, 2)})
    emo_ok = sum(1 for row in report["emotion"] if row["valid"])
    log(f"  {emo_ok}/{len(report['emotion'])} emotion clips valid")
    flush()

    # ------------------------------------------------------------------ judging
    log("")
    log("=== 6. judging (valid clips only) ===")
    del tts.lm
    torch.cuda.empty_cache()
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(ASR_ID, token=token)
    asr = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_ID, token=token, dtype=torch.float16).to("cuda").eval()

    def transcribe(wav: np.ndarray) -> str:
        import torchaudio

        resampled = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(wav)), SAMPLE_RATE, JUDGE_RATE).numpy()
        features = processor(resampled, sampling_rate=JUDGE_RATE,
                             return_tensors="pt").input_features.to("cuda").half()
        with torch.inference_mode():
            ids = asr.generate(features, language="ta", task="transcribe",
                               max_new_tokens=220)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    by_id = {row["id"]: row for row in subset}
    for name, out in rendered.items():
        per_clip, mine_counts, base_counts = [], [], []
        for clip_id, (wav, check, n_new) in out.items():
            row = by_id[clip_id]
            reference = clips[clip_id]["text"]
            entry = {"id": clip_id, "category": row["category"], "speaker": row["speaker"],
                     "valid": check.valid, "reason": check.reason,
                     "seconds": round(check.seconds, 2),
                     "level_dbfs": None if not np.isfinite(check.level_dbfs)
                     else round(check.level_dbfs, 1),
                     "new_tokens": n_new}
            theirs = cer(row["b7_hypothesis"], reference)
            entry["base_cer"] = round(100.0 * (theirs.rate or 0.0), 2)
            if check.valid:
                hypothesis = transcribe(wav)
                mine = cer(hypothesis, reference)  # (hypothesis, reference), in that order
                mine_counts.append(mine)
                base_counts.append(theirs)
                entry["geetha_cer"] = round(100.0 * (mine.rate or 0.0), 2)
                entry["hypothesis"] = hypothesis
            per_clip.append(entry)
        arm = report["arms"][name]
        arm["per_clip"] = per_clip
        arm["judged"] = len(mine_counts)
        if len(mine_counts) >= 2:
            ours_ci, base_ci = bootstrap_ci(mine_counts), bootstrap_ci(base_counts)
            paired = paired_bootstrap(mine_counts, base_counts)
            valid_rows = [e for e in per_clip if e["valid"]]
            arm.update(
                geetha_cer=round(100 * ours_ci.point, 2),
                geetha_ci=[round(100 * ours_ci.low, 2), round(100 * ours_ci.high, 2)],
                base_cer_same_clips=round(100 * base_ci.point, 2),
                base_ci=[round(100 * base_ci.low, 2), round(100 * base_ci.high, 2)],
                paired_delta=round(100 * paired.delta, 2),
                paired_delta_ci=[round(100 * paired.low, 2), round(100 * paired.high, 2)],
                paired_p_value=paired.p_value,
                better_than_base=sum(1 for e in valid_rows
                                     if e["geetha_cer"] < e["base_cer"]),
                median_seconds=float(np.median([e["seconds"] for e in valid_rows])))
            log(f"  {name}: {arm['validity']}")
            log(f"    CER {arm['geetha_cer']:.2f} {arm['geetha_ci']} vs base "
                f"{arm['base_cer_same_clips']:.2f} on the same {len(mine_counts)} clips; "
                f"paired {arm['paired_delta']:+.2f} {arm['paired_delta_ci']}, "
                f"p = {paired.p_value:.4f}; better on "
                f"{arm['better_than_base']}/{len(mine_counts)}")
        else:
            log(f"  {name}: fewer than two valid clips, no CER")
        flush()
    report["caveat"] = (
        "The base column is B7's stored hypothesis for the same prompt. B7 did not apply "
        "the loudness gate, so a base clip that was itself mute or ran away is still in "
        "the base pool; the pair is exact on prompts, seeds, judge and decoding, not on "
        "gating. B_rp11 changes the decoding and is therefore not like-for-like with B7.")
    flush()

    # ------------------------------------------------------------------ the listen
    headline = [e for e in report["arms"]["B_plain"].get("per_clip", []) if e["valid"]]
    headline.sort(key=lambda e: e["geetha_cer"])
    picks = headline[:5] + headline[-5:]
    lines = ["# What to listen to (M2 main run, Stage B adapter)", "",
             "## 1. Emotion: the same sentence, seven ways",
             "`clips/emo_geetha_c01_anitha_*.wav`, then `..._arun_*`, then carrier `c02`. "
             "Compare with the base model: `clips/emo_base_c01_*`. CONV is the neutral one.",
             "", "## 2. Breath: with and without the tag",
             "`clips/breath_stageB_anitha_<n>_with.wav` against `..._without.wav` "
             "(and `arun`). Is there an audible breath where the tag was?", "",
             "## 3. Ordinary sentences: five best and five worst by CER", ""]
    lines += [f"- `clips/{e['id']}.wav` ({e['category']}, {e['speaker']}, CER "
              f"{e['geetha_cer']} vs base {e['base_cer']}): heard as "
              f"“{e['hypothesis'][:90]}”" for e in picks]
    invalid = [e for e in report["arms"]["B_plain"].get("per_clip", []) if not e["valid"]]
    if invalid:
        lines += ["", "## 4. Clips the gate rejected (listen to confirm it was right)", ""]
        lines += [f"- `clips/{e['id']}.wav`: {e['reason']}" for e in invalid]
    (OUT_DIR / "LISTEN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("")
    log(f"wrote m2_main_eval.json and LISTEN.md; {len(list(audio_dir.iterdir()))} wavs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
