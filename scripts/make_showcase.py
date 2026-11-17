"""Write kaggle\\m3_showcase\\m3_showcase.py: several takes of the sentences we will play
on the public pages, so the operator can choose the take to show.

These are SELECTED samples and are labelled so wherever they appear; every unselected
benchmark clip is in eval/. Same kernel as the emotion listen (release files embedded
byte for byte), with a different clip list and one seed per take.

Usage:  python scripts\\make_showcase.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_emotion_listen as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "m3_showcase"
SLUG = "geetha-m3-showcase"

NEUTRAL = [
    "வணக்கம், என் பேரு கீதா. நான் தமிழ்ல பேசுற ஒரு குரல்.",
    "நாளைக்கு காலையில பத்து மணிக்கு ஒரு மீட்டிங் இருக்கு, மறக்காம வந்துடுங்க.",
    "இன்னைக்கு சென்னையில லேசான மழை பெய்யும், குடை எடுத்துட்டு போங்க.",
    "ஒரு ஊர்ல ஒரு பாட்டி இருந்தாங்க, அவங்க தினமும் காலையில வடை சுடுவாங்க.",
    "உங்க ஆர்டர் கிளம்பிடுச்சு, இன்னும் இருபது நிமிஷத்துல உங்க வீட்டுக்கு வந்துடும்.",
]
BREATH = [
    "[breath] சரி, நான் ஒண்ணு சொல்றேன், கவனமா கேளு.",
    "[breath] இத எப்படி சொல்றதுன்னே தெரியல, ஆனா சொல்லித்தான் ஆகணும்.",
]
STYLED = {
    "HAPPY": "எனக்கு வேலை கிடைச்சிருச்சு, ரொம்ப சந்தோஷமா இருக்கு!",
    "SAD": "அவங்க போனதுக்கு அப்புறம் வீடே வெறிச்சோடி இருக்கு.",
    "ANGER": "எத்தன தடவ சொல்றது, என் பொருள என்கிட்ட கேக்காம எடுக்காதன்னு!",
    "FEAR": "யாரோ கதவ தட்டுற மாதிரி இருக்கு, இந்த நேரத்துல யாரா இருக்கும்?",
    "SURPRISE": "அடடா, நீ எப்போ ஊருக்கு வந்த? சொல்லவே இல்லையே!",
    "DISGUST": "சீ, இந்த சாப்பாடு கெட்டுப்போன வாசன அடிக்குது, என்னால சாப்பிட முடியாது.",
}
NUMBERS = "இந்த புத்தகத்தோட விலை ₹350, 15/11/2026 வரைக்கும் 20% தள்ளுபடி இருக்கு."
LONG = ("நேத்து சாயங்காலம் நான் கடைத்தெருவுக்கு போனேன். அங்க ஒரு பழைய நண்பன பாத்தேன். "
        "ரொம்ப நாள் கழிச்சு பாத்ததால ரெண்டு பேரும் ஒரு டீ கடையில உக்காந்து பேசினோம். "
        "அடுத்த மாசம் ஊருக்கு வரும்போது வீட்டுக்கு வரேன்னு சொல்லிட்டு போனான்.")


def jobs() -> list[dict]:
    plan: list[tuple[str, str, str, str, int]] = []      # name, text, voice, style, takes
    for voice in ("anitha", "arun"):
        plan += [(f"neutral{i}_{voice}", text, voice, "CONV", 3) for i, text in enumerate(NEUTRAL)]
        plan += [(f"breath{i}_{voice}", text, voice, "CONV", 3) for i, text in enumerate(BREATH)]
        plan += [(f"numbers_{voice}", NUMBERS, voice, "CONV", 2), (f"long_{voice}", LONG, voice, "CONV", 1)]
    plan += [(f"{style.lower()}_anitha", text, "anitha", style, 3) for style, text in STYLED.items()]
    plan += [(f"{style.lower()}_arun", STYLED[style], "arun", style, 2)
             for style in ("HAPPY", "SAD", "SURPRISE")]
    out, number = [], 0
    for name, text, voice, style, takes in plan:
        for take in range(takes):
            out.append({"file": f"{name}_take{take + 1}.wav", "kind": name.split("_")[0].rstrip("0123456789"),
                        "name": name, "voice": voice, "target": style, "style": style,
                        "text": text, "seed_index": number})
            number += 1
    return out


def main() -> int:
    base.KERNEL, base.SLUG, base.jobs = KERNEL, SLUG, jobs
    base.main()
    built = KERNEL / "m3_emotion_listen.py"
    script = (built.read_text(encoding="utf-8")
              .replace("m3_emotion_listen.json", "m3_showcase.json")
              .replace("SEED = 1811", "SEED = 2011")
              .replace("M3: ear test for the six styles (styled against CONV, same sentence, same seed)\n"
                       "and the model card's own examples. Written by scripts\\make_emotion_listen.py.",
                       "M3: several takes of the sentences for the public pages; the operator picks.\n"
                       "Written by scripts\\make_showcase.py."))
    compile(script, "m3_showcase.py", "exec")
    built.unlink()
    (KERNEL / "m3_showcase.py").write_text(script, encoding="utf-8", newline="\n")
    meta = json.loads((KERNEL / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["code_file"] = "m3_showcase.py"
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(jobs())} clips -> {KERNEL / 'm3_showcase.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
