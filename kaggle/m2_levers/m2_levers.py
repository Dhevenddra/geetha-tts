r"""M2 Stage B: is the expression in the model and lost in decoding? (free Kaggle GPU)

On 11 Nov the operator listened to m2_main_eval and heard: the breath tag gives a pause,
not a breath; emotion is weak and close to the base model. Before any more training is
bought, three free decoding levers are tried on the same adapter:

  S0  temperature 0.6, no penalty          what every clip so far used
  S1  temperature 0.9, repetition 1.1      low temperature flattens prosody
  S2  S1 + style guidance 2.0              classifier-free guidance: the logits with the
  S3  S1 + style guidance 3.5              style block pushed away from those without it
                                           (D-051 trained with style dropout 0.1)

Each emotion is rendered on a neutral carrier and on a sentence whose words carry the
emotion, with a CONV render of the same text as the reference. Pitch and energy statistics
are written beside the clips so "did the style move the prosody" is a number as well as a
listen. Breath pairs get the same treatment with guidance on the tag.
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
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTERS = {"stageB": "m2-main/stageB/final"}
SAMPLE_RATE = 24_000
MAX_NEW_TOKENS = int(22.0 * 82)
HEAD = "<|start_of_human|><|begin_of_text|>"
TAIL = "<|eot_id|><|end_of_human|><|start_of_ai|>"
SEED = 4242
_MAX_MOUNT_DEPTH = 4

SETTINGS = {
    "S0": dict(temperature=0.6, repetition_penalty=1.0, guidance=0.0),
    "S1": dict(temperature=0.9, repetition_penalty=1.1, guidance=0.0),
    "S2": dict(temperature=0.9, repetition_penalty=1.1, guidance=2.0),
    "S3": dict(temperature=0.9, repetition_penalty=1.1, guidance=3.5),
}
NEUTRAL = "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு."
# Written for this run; not eval-vault items.
EMOTIONAL = {
    "HAPPY": "ஐயோ, நிஜமாவா! எனக்கு ரொம்ப சந்தோஷமா இருக்கு!",
    "SAD": "அவர் இனிமே வரமாட்டார்னு நினைச்சா மனசு ரொம்ப கஷ்டமா இருக்கு.",
    "ANGER": "எத்தனை தடவை சொல்லணும்? இனிமே இப்படி பண்ணாத!",
}
BREATH_TAG = "<|nv_breath|>"
BREATH = [
    "அது <|nv_breath|> ஒரு பெரிய விஷயம் இல்ல",
    "நான் <|nv_breath|> அங்கே போகலைன்னு சொன்னேன்",
    "சரி <|nv_breath|> நாளைக்கு பாக்கலாம்",
    "எனக்கு தெரியல <|nv_breath|> நீயே சொல்லு",
]


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



def prosody(wav: np.ndarray) -> dict:
    """Pitch and energy statistics: crude, but the same for every clip."""
    import librosa

    if len(wav) < SAMPLE_RATE // 4:
        return {}
    f0, _, _ = librosa.pyin(wav.astype(np.float32), fmin=70, fmax=500, sr=SAMPLE_RATE)
    f0 = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=wav.astype(np.float32))[0]
    if len(f0) < 5:
        return {}
    semitones = 12 * np.log2(f0 / np.median(f0))
    return {"f0_median_hz": round(float(np.median(f0)), 1),
            "f0_range_semitones": round(float(np.percentile(semitones, 95)
                                              - np.percentile(semitones, 5)), 2),
            "energy_db_std": round(float(np.std(20 * np.log10(rms + 1e-6))), 2),
            "seconds": round(len(wav) / SAMPLE_RATE, 2)}


def main() -> int:
    fix_environment()
    import soundfile as sf
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    token = require_token()
    ensure_dependencies()
    import_geetha()
    from geetha.eval.validity import check_clip

    adapters = fetch_adapters(token)
    tts = build_tts(token, adapters)
    from inference import ids_to_codes  # type: ignore

    tok = tts.tok
    start_speech = tok.convert_tokens_to_ids("<|start_of_speech|>")
    stop_ids = {tts._eos, tok.eos_token_id}

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def prompt(text: str, speaker: str, style: str = "") -> list[int]:
        return (enc(HEAD) + enc(f"<|speaker>{speaker}<speaker|>\n")
                + (enc(f"<|style>{style}<style|>\n") if style else [])
                + enc("<|env>studio<env|>\n") + enc(text) + enc(TAIL) + [start_speech])

    class Guidance(LogitsProcessor):
        """Classifier-free guidance against a second prompt, with its own cache."""

        def __init__(self, other: list[int], scale: float):
            self.other = torch.tensor([other], device=tts.device)
            self.scale, self.past, self.started = scale, None, False

        def __call__(self, input_ids, scores):
            feed = input_ids[:, -1:] if self.started else self.other
            out = tts.lm(input_ids=feed, past_key_values=self.past, use_cache=True)
            self.past, self.started = out.past_key_values, True
            cond = torch.log_softmax(scores.float(), dim=-1)
            unc = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
            return (unc + self.scale * (cond - unc)).to(scores.dtype)

    def generate(ids: list[int], other: list[int] | None, setting: dict, seed: int):
        torch.manual_seed(seed)
        x = torch.tensor([ids], device=tts.device)
        processors = LogitsProcessorList()
        if other is not None and setting["guidance"] > 0:
            processors.append(Guidance(other, setting["guidance"]))
        with torch.no_grad():
            gen = tts.lm.generate(
                input_ids=x, attention_mask=torch.ones_like(x),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=setting["temperature"], top_p=0.9, top_k=50,
                repetition_penalty=setting["repetition_penalty"],
                eos_token_id=[tts._eos, tok.eos_token_id],
                pad_token_id=tok.eos_token_id, logits_processor=processors)
            new = gen[0].tolist()[len(ids):]
            ended = bool(new) and new[-1] in stop_ids
            try:
                codes = ids_to_codes(new, tts._base, tts.device)
                wav = tts.vocos(tts.snac.quantizer.from_codes(codes).float())
                wav = wav[0, 0].clamp(-1, 1).float().cpu().numpy()
            except ValueError:
                wav = np.zeros(0, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32).reshape(-1), ended

    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    rows = []

    def flush() -> None:
        (OUT_DIR / "m2_levers.json").write_text(
            json.dumps({"settings": SETTINGS, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    started = time.time()
    texts = {"neutral": {e: NEUTRAL for e in EMOTIONAL}, "emo": EMOTIONAL}
    for voice in ("anitha", "arun"):
        for kind, by_emotion in texts.items():
            for emotion, text in by_emotion.items():
                plain = prompt(text, voice)
                for style in ("CONV", emotion):
                    if style == "CONV" and kind == "neutral" and emotion != "HAPPY":
                        continue  # one CONV reference of the neutral carrier is enough
                    for name, setting in SETTINGS.items():
                        wav, ended = generate(prompt(text, voice, style), plain, setting, SEED)
                        check = check_clip(wav, SAMPLE_RATE, ended_itself=ended)
                        file = f"lev_{voice}_{kind}_{emotion}_{style}_{name}.wav"
                        if len(wav):
                            sf.write(audio / file, wav, SAMPLE_RATE)
                        rows.append({"file": file, "voice": voice, "text_kind": kind,
                                     "emotion": emotion, "style": style, "setting": name,
                                     "valid": check.valid, "reason": check.reason,
                                     **(prosody(wav) if check.valid else {})})
                log(f"  {voice} {kind} {emotion}: {len(rows)} clips, "
                    f"{time.time() - started:.0f} s")
                flush()

    for index, marked in enumerate(BREATH):
        without = marked.replace(BREATH_TAG, " ").replace("  ", " ").strip()
        for name in ("S1", "S3"):
            setting = SETTINGS[name]
            wav, ended = generate(prompt(marked, "anitha", "CONV"),
                                  prompt(without, "anitha", "CONV"), setting, SEED + index)
            check = check_clip(wav, SAMPLE_RATE, ended_itself=ended)
            file = f"breath_{index}_tag_{name}.wav"
            if len(wav):
                sf.write(audio / file, wav, SAMPLE_RATE)
            rows.append({"file": file, "voice": "anitha", "text_kind": "breath",
                         "setting": name, "valid": check.valid, "reason": check.reason,
                         "seconds": round(len(wav) / SAMPLE_RATE, 2)})
    flush()

    log("")
    log("=== does the style move the prosody? (emotion minus CONV, same text and setting) ===")
    index = {(r["voice"], r["text_kind"], r["emotion"], r["style"], r["setting"]): r
             for r in rows if r["text_kind"] != "breath"}
    for name in SETTINGS:
        deltas, unusable = [], 0
        for (voice, kind, emotion, style, setting), row in index.items():
            if setting != name or style == "CONV":
                continue
            ref = index.get((voice, kind, emotion if kind == "emo" else "HAPPY", "CONV", name))
            if not row.get("f0_range_semitones") or not ref or not ref.get("f0_range_semitones"):
                unusable += 1
                continue
            deltas.append((abs(row["f0_median_hz"] / ref["f0_median_hz"] - 1) * 100,
                           row["f0_range_semitones"] - ref["f0_range_semitones"],
                           row["energy_db_std"] - ref["energy_db_std"]))
        if deltas:
            d = np.array(deltas)
            log(f"  {name}: n={len(d)} (unusable {unusable})  |median pitch shift| "
                f"{d[:, 0].mean():.1f}%  pitch range {d[:, 1].mean():+.2f} st  "
                f"energy spread {d[:, 2].mean():+.2f} dB")
    bad = [r["file"] for r in rows if not r["valid"]]
    log(f"invalid clips: {len(bad)}/{len(rows)} {bad}")
    log(f"wrote m2_levers.json and {len(list(audio.iterdir()))} wavs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
