"""Write kaggle\\m3_emotion_listen\\m3_emotion_listen.py: an ear test for the six styles
and the model card's own examples.

The kernel is the mid-split kernel (same loading, same release files embedded byte for
byte) with a different list of clips:

  emo   12 sentences, 2 per style, each worded to carry its feeling. Per voice each is
        rendered twice with the same seed: in its style, and as CONV. The listener
        hears the pair blind and says which one sounds more <feeling>.
  card  the examples printed in release\\README.md, so the page can play what it shows.

Usage:  python scripts\\make_emotion_listen.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
SOURCE = ROOT / "kaggle" / "m3_mid_split" / "template.py"
KERNEL = ROOT / "kaggle" / "m3_emotion_listen"
SLUG = "geetha-m3-emotion-listen"
FILES = ("inference.py", "tamil_text.py", "breaths.json")

EMOTION = {
    "HAPPY": ["எனக்கு வேலை கிடைச்சிருச்சு, ரொம்ப சந்தோஷமா இருக்கு!",
              "இன்னைக்கு என் தங்கச்சிக்கு குழந்தை பிறந்திருக்கு, எல்லாரும் ரொம்ப குஷியா இருக்கோம்!"],
    "SAD": ["அவங்க போனதுக்கு அப்புறம் வீடே வெறிச்சோடி இருக்கு.",
            "எவ்வளவு முயற்சி பண்ணியும் இந்த தடவையும் எனக்கு கிடைக்கல, மனசே சரியில்ல."],
    "ANGER": ["எத்தன தடவ சொல்றது, என் பொருள என்கிட்ட கேக்காம எடுக்காதன்னு!",
              "நீ சொன்ன நேரத்துக்கு வரல, ஒரு ஃபோன் கூட பண்ணல, இது ரொம்ப தப்பு!"],
    "FEAR": ["யாரோ கதவ தட்டுற மாதிரி இருக்கு, இந்த நேரத்துல யாரா இருக்கும்?",
             "எனக்கு ரொம்ப பயமா இருக்கு, தயவுசெஞ்சு என்ன தனியா விட்டுட்டு போகாத."],
    "SURPRISE": ["என்னது, நீயா இத பண்ணின? நான் நம்பவே முடியல!",
                 "அடடா, நீ எப்போ ஊருக்கு வந்த? சொல்லவே இல்லையே!"],
    "DISGUST": ["சீ, இந்த சாப்பாடு கெட்டுப்போன வாசன அடிக்குது, என்னால சாப்பிட முடியாது.",
                "இவ்வளவு அழுக்கான இடத்துல எப்படி தான் இருக்காங்களோ, பாக்கவே அருவருப்பா இருக்கு."],
}
CARD = [
    ("card_anitha_neutral", "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", "anitha", "CONV"),
    ("card_arun_neutral", "நாளைக்கு காலையில ஒரு மீட்டிங் இருக்கு.", "arun", "CONV"),
    ("card_breath", "[breath] சரி, நான் ஒண்ணு சொல்றேன், கவனமா கேளு.", "arun", "CONV"),
    ("card_numbers", "விலை ₹350, 15/11/2026 வரைக்கும் 20% தள்ளுபடி.", "anitha", "CONV"),
    ("card_sad_breath", "[breath] இத பத்தி நாம நாளைக்கு பேசலாம்.", "arun", "SAD"),
]

LOOP = '''    for index, job in enumerate(JOBS):
        file = job["file"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wav = tts.speak(job["text"], speaker=job["voice"], style=job["style"],
                            seed=SEED + 10 * job["seed_index"], retries=0)
        invalid = [str(w.message) for w in caught
                   if issubclass(w.category, RuntimeWarning)
                   and "stayed invalid" in str(w.message)]
        row = dict(job, valid=not invalid and len(wav) > 0, reason="; ".join(invalid),
                   seconds=round(len(wav) / SAMPLE_RATE, 2))
        if len(wav):
            sf.write(audio / file, wav, SAMPLE_RATE)
        rows.append(row)
        log(f"  {file}: valid={row['valid']} {row['seconds']} s "
            f"[{time.time() - started:.0f} s total]")
        (OUT_DIR / "m3_emotion_listen.json").write_text(
            json.dumps({"seed": SEED, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8")
'''


def jobs() -> list[dict]:
    out, number = [], 0
    for voice in ("anitha", "arun"):
        for style, sentences in EMOTION.items():
            for which, text in enumerate(sentences):
                for arm in (style, "CONV"):      # same seed for the styled and the plain take
                    out.append({"file": f"emo_{voice}_{style.lower()}{which}_{'styled' if arm == style else 'plain'}.wav",
                                "kind": "emo", "voice": voice, "target": style, "style": arm,
                                "text": text, "seed_index": number})
                number += 1
    for name, text, voice, style in CARD:
        out.append({"file": f"{name}.wav", "kind": "card", "voice": voice, "target": style,
                    "style": style, "text": text, "seed_index": number})
        number += 1
    return out


def main() -> int:
    template = SOURCE.read_text(encoding="utf-8")
    head_end = template.index('"""', 5) + 3
    block_start, block_end = template.index("ARMS = ("), template.index("def log(")
    loop_start = template.index("    for voice in inference.VOICES:")
    loop_end = template.index('    bad = [r["file"]')
    payload = {name: base64.b64encode((RELEASE / name).read_bytes()).decode("ascii")
               for name in FILES}
    plan = jobs()
    script = ('r"""M3: ear test for the six styles (styled against CONV, same sentence, same seed)\n'
              'and the model card\'s own examples. Written by scripts\\make_emotion_listen.py.\n'
              'Single shot. Audio is private to this kernel (D-007).\n"""'
              + template[head_end:block_start]
              + "PAYLOAD = " + json.dumps(payload, indent=1) + "\n\n"
              + "JOBS = json.loads(r'''" + json.dumps(plan, ensure_ascii=False) + "''')\n\n\n"
              + template[block_end:loop_start] + LOOP + template[loop_end:])
    script = script.replace("SEED = 1711", "SEED = 1811")
    compile(script, "m3_emotion_listen.py", "exec")
    KERNEL.mkdir(parents=True, exist_ok=True)
    (KERNEL / "m3_emotion_listen.py").write_text(script, encoding="utf-8", newline="\n")
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": f"dhevenddra/{SLUG}", "title": SLUG,
        "code_file": "m3_emotion_listen.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["dhevenddra/geetha-secrets"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2),
        encoding="utf-8")
    print(f"{len(plan)} clips -> {KERNEL / 'm3_emotion_listen.py'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
