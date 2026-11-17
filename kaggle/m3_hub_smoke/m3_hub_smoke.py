r"""M3: does the model card's "Use" section work, starting from nothing but the Hub?

Follows the card: install the listed packages, download inference.py, tamil_text.py and
breaths.json from the model repo, then Geetha("Dhevenddra/geetha-tts"). The repo is
still private, so the read token stands in for `hf auth login`. Renders the card's own
examples once each. Audio is private to this kernel.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "Dhevenddra/geetha-tts"
OUT = Path("/kaggle/working")
EXAMPLES = [
    ("anitha_neutral", "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", "anitha", "CONV"),
    ("arun_breath", "[breath] சரி, நான் ஒண்ணு சொல்றேன், கவனமா கேளு.", "arun", "CONV"),
    ("anitha_happy", "எனக்கு வேலை கிடைச்சிருச்சு, ரொம்ப சந்தோஷமா இருக்கு!", "anitha", "HAPPY"),
    ("anitha_numbers", "விலை ₹350, 15/11/2026 வரைக்கும் 20% தள்ளுபடி.", "anitha", "CONV"),
]


def main() -> int:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   capture_output=True, text=True)        # Kaggle-only: it breaks PEFT
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snac", "soundfile"], check=True)
    from huggingface_hub import hf_hub_download

    token = next(Path("/kaggle/input").rglob("hf_read_token.txt")).read_text(encoding="utf-8").strip()
    folder = Path("/kaggle/tmp/geetha")
    for name in ("inference.py", "tamil_text.py", "breaths.json"):
        hf_hub_download(REPO, name, token=token, local_dir=folder)
    print("downloaded:", sorted(p.name for p in folder.iterdir() if p.is_file()), flush=True)
    os.chdir(folder)
    sys.path.insert(0, str(folder))
    from inference import Geetha

    started = time.time()
    tts = Geetha(REPO, token=token)
    print(f"loaded in {time.time() - started:.0f} s; breaths: "
          f"{ {v: len(b) for v, b in tts._breaths.items()} }", flush=True)
    rows = []
    for name, text, voice, style in EXAMPLES:
        clock = time.time()
        wav = tts.speak(text, speaker=voice, style=style, seed=7)
        tts.save(str(OUT / f"{name}.wav"), wav)
        rows.append({"name": name, "seconds": round(len(wav) / 24000, 2),
                     "render_s": round(time.time() - clock, 1)})
        print(rows[-1], flush=True)
    (OUT / "m3_hub_smoke.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ok = all(row["seconds"] > 0.5 for row in rows)
    print("RESULT:", "the card's steps work" if ok else "a clip came out empty", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
