r"""M3: does the file we SHIP work? release\inference.py, run exactly as a user would.

Every earlier breath result came from a hand-written sampling loop inside a kernel
(frame grammar, temperature 0.9). The release file does the insertion through
``generate`` with a logits processor, at the release decoding (temperature 0.6 for
neutral speech). That path has never run on a GPU. This kernel unpacks the release
files byte for byte (embedded below by scripts\make_release_smoke.py), imports them as
``inference`` the way the README says, and renders per voice:

  plain   4 sentences, no tag
  lead    the same 4 with [breath] before the first word
  mid     the same 4 with [breath] at the first clause break
  num     3 sentences written with digits, dates, rupees (read as words)
  long    1 text long enough to be split into several pieces
  sad     1 sentence, style SAD (guidance on) with a lead breath

All single-shot (retries=0), fixed seed. Reported: validity, breaths by the D-030
instrument and where they sit. The verdict on sound is the operator's ear.
Audio is private to this kernel (D-007).
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "bodhan-ai/indic-speak"
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTER = "m2-main/stageB/final"
SAMPLE_RATE = 24_000
SEED = 1418
_MAX_MOUNT_DEPTH = 4

__DETECTOR__

PAYLOAD = __PAYLOAD__

PLAIN = [
    "நேத்து ராத்திரி ரொம்ப நேரம் வேலை பாத்துட்டு இருந்தேன், அதனால காலையில எந்திரிக்கவே முடியல, ஆபீஸுக்கு லேட்டா தான் போனேன்",
    "எங்க ஊர்ல மழை பெய்ய ஆரம்பிச்சா, ரெண்டு மூணு நாளைக்கு நிக்கவே நிக்காது, அப்போ எல்லாரும் வீட்டுக்குள்ளயே இருப்பாங்க",
    "சரி நான் ஒண்ணு சொல்றேன் கவனமா கேளு, இந்த விஷயத்த இப்போதைக்கு யார்கிட்டயும் சொல்ல வேண்டாம்",
    "இன்னைக்கு காலையில பஸ்ல வரும்போது ஒரு பழைய நண்பன பாத்தேன், ரொம்ப வருஷம் கழிச்சு பேசினோம், மனசுக்கு ரொம்ப சந்தோஷமா இருந்துச்சு",
]
NUMBERS = [
    "இந்த புத்தகத்தோட விலை ₹350 தான், ஆனா 15/11/2026 வரைக்கும் 20% தள்ளுபடி இருக்கு.",
    "ரயில் காலை 6:45 க்கு கிளம்பும், மொத்தம் 1,250 பேர் பயணம் பண்றாங்க.",
    "2026-ல் எங்க ஊர்ல 3 புது பள்ளிக்கூடம் திறந்தாங்க, அதுல 5-ஆம் வகுப்பு வரைக்கும் இருக்கு.",
]
LONG = ("நேத்து சாயங்காலம் நான் கடைத்தெருவுக்கு போனேன். அங்க ஒரு பழைய நண்பன பாத்தேன். "
        "ரொம்ப நாள் கழிச்சு பாத்ததால ரெண்டு பேரும் ஒரு டீ கடையில உக்காந்து பேசினோம். "
        "அவன் இப்போ கோயம்புத்தூர்ல வேலை பாக்கறான்னு சொன்னான். "
        "அடுத்த மாசம் ஊருக்கு வரும்போது வீட்டுக்கு வரேன்னு சொல்லிட்டு போனான். "
        "வீட்டுக்கு வந்து அம்மாகிட்ட சொன்னதும் அவங்களுக்கும் ரொம்ப சந்தோஷம்.")
SAD = "[breath] இத பத்தி நாம நாளைக்கு பொறுமையா பேசலாம், இப்போ எனக்கு கொஞ்சம் ஓய்வு வேணும்"


def log(message: str) -> None:
    print(message, flush=True)


def fix_environment() -> None:
    """Kaggle's torchao 0.10.0 makes PEFT unable to build any LoRA; remove it first."""
    import subprocess

    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            capture_output=True, text=True)
    log(f"  pip uninstall torchao -> {result.returncode}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "snac==1.2.1", "soundfile==0.14.0"], check=True)


def _directories(root: Path, depth: int = 0):
    if not root.exists() or depth > _MAX_MOUNT_DEPTH:
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry
            yield from _directories(entry, depth + 1)


def find_dataset(*markers: str) -> Path:
    for mount in sorted(_directories(Path("/kaggle/input"))):
        if all((mount / marker).exists() for marker in markers):
            return mount
    raise SystemExit(f"could not find a mount containing {markers}")


def import_geetha() -> None:
    source = find_dataset("scoring", "text", "eval")
    root = Path("/kaggle/tmp/pkg")
    package = root / "geetha"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, package, dirs_exist_ok=True)
    sys.path.insert(0, str(root))


def unpack_release() -> Path:
    folder = Path("/kaggle/tmp/release")
    folder.mkdir(parents=True, exist_ok=True)
    for name, encoded in PAYLOAD.items():
        (folder / name).write_bytes(base64.b64decode(encoded))
        log(f"  unpacked {name}: {(folder / name).stat().st_size:,} bytes")
    return folder


def with_mid_tag(text: str) -> str:
    head, comma, tail = text.partition(", ")
    return f"{head}, [breath] {tail}" if comma else text


def main() -> int:
    fix_environment()
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    token = (find_dataset("hf_read_token.txt") / "hf_read_token.txt").read_text(
        encoding="utf-8").strip()
    base_dir = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)
    ckpt = Path(snapshot_download(CKPT_REPO, token=token, allow_patterns=[f"{ADAPTER}/*"]))
    adapter_dir = ckpt / ADAPTER
    if not (adapter_dir / "adapter_config.json").exists():
        adapter_dir = sorted(adapter_dir.rglob("adapter_config.json"))[0].parent
    token = ""                                   # third-party code is imported from here on
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)

    import_geetha()
    from geetha.audio.breath import BreathDetector
    from geetha.eval.breath_score import score_wave

    detector = BreathDetector.from_dict(DETECTOR)
    release = unpack_release()
    sys.path.insert(0, str(release))
    import inference                              # exactly what the README tells a user to do

    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    tts = inference.Geetha(str(adapter_dir), base=base_dir)
    library = {voice: len(items) for voice, items in tts._breaths.items()}
    log(f"  breath library: {library}")
    if set(library) != set(inference.VOICES):
        raise SystemExit("breaths.json did not load for both voices")

    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()

    def render(file: str, text: str, voice: str, arm: str, index: int, **kw) -> None:
        clock = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wav = tts.speak(text, speaker=voice, seed=SEED + 10 * index, retries=0, **kw)
        invalid = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
                   and "stayed invalid" in str(w.message)]
        row = {"file": file, "voice": voice, "arm": arm, "text": text,
               "sent": inference.verbalise(text) if kw.get("numbers", True) else text,
               "valid": not invalid and len(wav) > 0, "reason": "; ".join(invalid),
               "seconds": round(len(wav) / SAMPLE_RATE, 2),
               "render_s": round(time.time() - clock, 1)}
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
            level = 20 * np.log10(float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))) + 1e-9)
            row["level_dbfs"] = round(level, 1)
            scored = score_wave(wav, SAMPLE_RATE, detector, THRESHOLD)
            hits = [p for p in scored["pauses"] if p["breath"]]
            row.update(breaths=len(hits), max_score=scored["max_score"],
                       breath_at_s=[p["at_s"] for p in hits], pauses=scored["pauses"])
        rows.append(row)
        log(f"  {file}: valid={row['valid']} {row['seconds']} s  breaths={row.get('breaths')} "
            f"at {row.get('breath_at_s')}  max score {row.get('max_score')}  "
            f"[{row['render_s']} s to render, {time.time() - started:.0f} s total]")
        (OUT_DIR / "m3_release_smoke.json").write_text(
            json.dumps({"seed": SEED, "detector_threshold": THRESHOLD, "rows": rows},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    for voice in inference.VOICES:
        log(f"=== {voice} ===")
        for index, text in enumerate(PLAIN):
            render(f"{voice}_t{index}_plain.wav", text, voice, "plain", index)
            render(f"{voice}_t{index}_lead.wav", f"[breath] {text}", voice, "lead", index)
            render(f"{voice}_t{index}_mid.wav", with_mid_tag(text), voice, "mid", index)
        for index, text in enumerate(NUMBERS):
            render(f"{voice}_n{index}_num.wav", text, voice, "num", 20 + index)
        render(f"{voice}_long.wav", LONG, voice, "long", 30)
        render(f"{voice}_sad_lead.wav", SAD, voice, "sad", 31, style="SAD")

    log("")
    log("=== summary ===")
    for voice in inference.VOICES:
        for arm in ("plain", "lead", "mid", "num", "long", "sad"):
            group = [r for r in rows if r["voice"] == voice and r["arm"] == arm]
            good = [r for r in group if r["valid"]]
            early = sum(any(at <= 0.6 for at in r.get("breath_at_s", [])) for r in good)
            log(f"  {voice:7s} {arm:5s} valid {len(good)}/{len(group)}  "
                f"clips with a breath {sum(r.get('breaths', 0) > 0 for r in good)}  "
                f"breath in the first 0.6 s {early}  "
                f"mean seconds {np.mean([r['seconds'] for r in good]) if good else 0:.1f}")
    bad = [r["file"] for r in rows if not r["valid"]]
    log(f"invalid clips: {len(bad)}/{len(rows)} {bad}")
    log(f"wrote m3_release_smoke.json and {len(list(audio.iterdir()))} wavs "
        f"in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
