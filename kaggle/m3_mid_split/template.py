r"""M3: a breath in the MIDDLE of a sentence, old way against the new cut-at-the-tag way.

The operator's ear on 15 Nov: a mid-sentence breath often sounds pasted in. The release
file now has a second way to do it. Per voice, 6 sentences with [breath] written at the
first clause break, rendered three times with the same seed by release\inference.py
(embedded byte for byte by scripts\make_mid_split.py):

  none    mid_breath=False    the tag is ignored: the sentence with no breath
  old     mid_breath=True     the breath goes in at the speaker's first pause
  split   mid_breath="split"  the sentence is cut at the tag; the second half opens
                              with a breath and the two takes are joined

Single shot (retries=0). The verdict is the operator's ear; nothing is scored here
except validity. Audio is private to this kernel (D-007).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

OUT_DIR = Path("/kaggle/working")
MODEL_ID = "bodhan-ai/indic-speak"
BASE_REVISION = "76e0814f189321efe550011850cd288dae366aad"
CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
ADAPTER = "m2-main/stageB/final"
SAMPLE_RATE = 24_000
SEED = 1711
ARMS = (("none", False), ("old", True), ("split", "split"))

PAYLOAD = __PAYLOAD__

SENTENCES = [
    "சரி நான் ஒண்ணு சொல்றேன் கவனமா கேளு, [breath] இந்த விஷயத்த இப்போதைக்கு யார்கிட்டயும் சொல்ல வேண்டாம்",
    "நேத்து ராத்திரி ரொம்ப நேரம் வேலை பாத்துட்டு இருந்தேன், [breath] அதனால காலையில எந்திரிக்கவே முடியல",
    "இன்னைக்கு காலையில பஸ்ல வரும்போது ஒரு பழைய நண்பன பாத்தேன், [breath] ரொம்ப வருஷம் கழிச்சு பேசினோம்",
    "எனக்கு இந்த வேலை ரொம்ப பிடிச்சிருக்கு, [breath] ஆனா தினமும் இவ்வளவு தூரம் போயிட்டு வர்றது கஷ்டமா இருக்கு",
    "அம்மா சொன்ன மாதிரியே எல்லாம் நடந்துச்சு, [breath] நான் தான் அத அப்போ நம்பாம விட்டுட்டேன்",
    "மழை நின்னதும் நாம கிளம்பலாம், [breath] அதுவரைக்கும் இங்கயே உக்காந்து ஒரு டீ குடிக்கலாம்",
]


def log(message: str) -> None:
    print(message, flush=True)


def find_file(name: str) -> Path:
    for path in sorted(Path("/kaggle/input").rglob(name)):
        return path
    raise SystemExit(f"could not find {name} under /kaggle/input")


def main() -> int:
    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                            capture_output=True, text=True)   # it breaks PEFT on Kaggle
    log(f"  pip uninstall torchao -> {result.returncode}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "snac==1.2.1", "soundfile==0.14.0"], check=True)
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    token = find_file("hf_read_token.txt").read_text(encoding="utf-8").strip()
    base_dir = snapshot_download(MODEL_ID, revision=BASE_REVISION, token=token)
    ckpt = Path(snapshot_download(CKPT_REPO, token=token, allow_patterns=[f"{ADAPTER}/*"]))
    adapter_dir = ckpt / ADAPTER
    if not (adapter_dir / "adapter_config.json").exists():
        adapter_dir = sorted(adapter_dir.rglob("adapter_config.json"))[0].parent
    token = ""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(name, None)

    release = Path("/kaggle/tmp/release")
    release.mkdir(parents=True, exist_ok=True)
    for name, encoded in PAYLOAD.items():
        (release / name).write_bytes(base64.b64decode(encoded))
    sys.path.insert(0, str(release))
    import inference

    log(f"  GPU: {torch.cuda.get_device_name(0)}")
    tts = inference.Geetha(str(adapter_dir), base=base_dir)
    if set(tts._breaths) != set(inference.VOICES):
        raise SystemExit("breaths.json did not load for both voices")

    audio = OUT_DIR / "clips"
    audio.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()
    for voice in inference.VOICES:
        for index, text in enumerate(SENTENCES):
            for arm, setting in ARMS:
                file = f"{voice}_s{index}_{arm}.wav"
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    wav = tts.speak(text, speaker=voice, seed=SEED + 10 * index, retries=0,
                                    mid_breath=setting)
                invalid = [str(w.message) for w in caught
                           if issubclass(w.category, RuntimeWarning)
                           and "stayed invalid" in str(w.message)]
                row = {"file": file, "voice": voice, "sentence": index, "arm": arm,
                       "text": text, "valid": not invalid and len(wav) > 0,
                       "reason": "; ".join(invalid),
                       "seconds": round(len(wav) / SAMPLE_RATE, 2)}
                if len(wav):
                    sf.write(audio / file, wav, SAMPLE_RATE)
                rows.append(row)
                log(f"  {file}: valid={row['valid']} {row['seconds']} s "
                    f"[{time.time() - started:.0f} s total]")
                (OUT_DIR / "m3_mid_split.json").write_text(
                    json.dumps({"seed": SEED, "rows": rows}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    bad = [r["file"] for r in rows if not r["valid"]]
    log(f"invalid clips: {len(bad)}/{len(rows)} {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
