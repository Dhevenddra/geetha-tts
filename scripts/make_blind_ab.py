"""Blind A/B listening sheet: base vs Geetha (D-058), plus the Stage C tag check for arun.

    make   copies the clips under neutral names, level-matched, writes blind\\listen.html
           and keeps the answer key OUTSIDE the folder the listener opens.
    score  reads the answers file the page downloads and prints the tallies.

Usage:
    python scripts\\make_blind_ab.py make
    python scripts\\make_blind_ab.py score E:\\path\\to\\blind_answers.json
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RUN = Path(r"E:\Geetha\runs\m3_base_vs_geetha")
STAGEC = Path(r"E:\Geetha\runs\m3_breath_stagec_v3")
OUT = RUN / "blind"
KEY = RUN / "blind_key.json"          # not inside OUT: the listener never sees it
SEED = 2711
TARGET_RMS_DB = -23.0

# Same sentences the kernels spoke (t0..t7), breath marks removed for display.
TEXTS = [
    "நேத்து ராத்திரி ரொம்ப நேரம் வேலை பாத்துட்டு இருந்தேன், அதனால காலையில எந்திரிக்கவே முடியல, ஆபீஸுக்கு லேட்டா தான் போனேன்",
    "எங்க ஊர்ல மழை பெய்ய ஆரம்பிச்சா, ரெண்டு மூணு நாளைக்கு நிக்கவே நிக்காது, அப்போ எல்லாரும் வீட்டுக்குள்ளயே இருப்பாங்க",
    "சரி நான் ஒண்ணு சொல்றேன் கவனமா கேளு, இந்த விஷயத்த இப்போதைக்கு யார்கிட்டயும் சொல்ல வேண்டாம்",
    "இன்னைக்கு காலையில பஸ்ல வரும்போது ஒரு பழைய நண்பன பாத்தேன், ரொம்ப வருஷம் கழிச்சு பேசினோம், மனசுக்கு ரொம்ப சந்தோஷமா இருந்துச்சு",
    "அம்மா சொன்னாங்க சீக்கிரம் வீட்டுக்கு வான்னு, ஆனா வேலை முடியவே ராத்திரி ஆயிடுச்சு",
    "உண்மைய சொல்லணும்னா எனக்கு அந்த படம் அவ்வளவா பிடிக்கல, ஆனா பாட்டு மட்டும் நல்லா இருந்துச்சு",
    "கடைக்கு போய் காய்கறி வாங்கிட்டு வரலாம்னு நினைச்சேன், அதுக்குள்ள மழை வந்துடுச்சு",
    "இத பத்தி நாம நாளைக்கு பொறுமையா பேசலாம், இப்போ எனக்கு கொஞ்சம் ஓய்வு வேணும்",
]

SECTIONS = {
    "plain": {
        "title": "Part 1 - plain speech (16 pairs)",
        "ask": "Which one sounds more like a real person talking?",
        "options": [("A", "A"), ("B", "B"), ("same", "No real difference")],
    },
    "breath": {
        "title": "Part 2 - with a breath put in (16 pairs)",
        "ask": "Which one sounds more natural, breath included?",
        "options": [("A", "A"), ("B", "B"), ("same", "No real difference")],
    },
    "tag": {
        "title": "Part 3 - arun only (4 pairs)",
        "ask": "In which one do you hear him breathe?",
        "options": [("A", "A"), ("B", "B"), ("both", "Both"), ("neither", "Neither")],
    },
}


def level(src: Path, dst: Path) -> None:
    audio, rate = sf.read(src, dtype="float32")
    rms = float(np.sqrt(np.mean(audio ** 2))) or 1e-9
    gain = (10 ** (TARGET_RMS_DB / 20)) / rms
    peak = float(np.max(np.abs(audio))) or 1e-9
    gain = min(gain, 0.98 / peak)
    sf.write(dst, audio * gain, rate, subtype="PCM_16")


def collect() -> list[dict]:
    pairs = []
    clips = RUN / "clips"
    for path in sorted(clips.glob("*_base_*.wav")):
        match = re.fullmatch(r"(anitha|arun)_t(\d)_base_(plain|breath_lead|breath_mid)", path.stem)
        if not match:
            continue
        voice, text, arm = match.group(1), int(match.group(2)), match.group(3)
        other = clips / f"{voice}_t{text}_geetha_{arm}.wav"
        if not other.exists():
            raise SystemExit(f"missing partner for {path.name}")
        pairs.append({"section": "plain" if arm == "plain" else "breath", "voice": voice,
                      "text": text, "arm": arm,
                      "x": ("base", path), "y": ("geetha", other)})
    for text in range(4):
        tag = STAGEC / "clips" / f"arun_t{text}_C_tag.wav"
        notag = STAGEC / "clips" / f"arun_t{text}_C_notag.wav"
        if not (tag.exists() and notag.exists()):
            raise SystemExit(f"missing Stage C clips for arun t{text}")
        pairs.append({"section": "tag", "voice": "arun", "text": text, "arm": "stageC",
                      "x": ("C_notag", notag), "y": ("C_tag", tag)})
    return pairs


def make() -> None:
    rng = random.Random(SEED)
    pairs = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("p*_[AB].wav"):
        old.unlink()
    key, blocks, number = {}, [], 0
    for name, section in SECTIONS.items():
        mine = [p for p in pairs if p["section"] == name]
        rng.shuffle(mine)
        # half the pairs put the second system on side A, so a side habit cannot favour either
        flips = [i % 2 == 0 for i in range(len(mine))]
        rng.shuffle(flips)
        blocks.append(f"<h2>{html.escape(section['title'])}</h2>"
                      f"<p class='ask'>{html.escape(section['ask'])}</p>")
        for pair, flip in zip(mine, flips):
            number += 1
            pid = f"p{number:02d}"
            a, b = (pair["y"], pair["x"]) if flip else (pair["x"], pair["y"])
            level(a[1], OUT / f"{pid}_A.wav")
            level(b[1], OUT / f"{pid}_B.wav")
            key[pid] = {"section": name, "voice": pair["voice"], "text": pair["text"],
                        "arm": pair["arm"], "A": a[0], "B": b[0],
                        "A_src": str(a[1]), "B_src": str(b[1])}
            radios = "".join(
                f"<label><input type='radio' name='{pid}' value='{value}'> {html.escape(label)}</label>"
                for value, label in section["options"])
            blocks.append(
                f"<div class='pair' id='{pid}'><div class='num'>{number}</div>"
                f"<div class='body'><div class='text'>{html.escape(TEXTS[pair['text']])}</div>"
                f"<div class='players'><span>A</span><audio controls preload='none' src='{pid}_A.wav'></audio>"
                f"<span>B</span><audio controls preload='none' src='{pid}_B.wav'></audio></div>"
                f"<div class='pick'>{radios}</div>"
                f"<input class='note' type='text' data-note='{pid}' placeholder='note (optional)'></div></div>")
    KEY.write_text(json.dumps({"seed": SEED, "target_rms_db": TARGET_RMS_DB, "pairs": key},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "listen.html").write_text(PAGE.replace("@@BLOCKS@@", "\n".join(blocks))
                                     .replace("@@TOTAL@@", str(number)), encoding="utf-8")
    print(f"{number} pairs -> {OUT / 'listen.html'}")
    print(f"key (do not open before listening) -> {KEY}")


def wilson(wins: int, total: int) -> str:
    if total == 0:
        return "n/a"
    z, p = 1.96, wins / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return f"{100 * p:.0f}% [{100 * (centre - half):.0f}, {100 * (centre + half):.0f}]"


def score(answers_path: Path) -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))["pairs"]
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    answers, notes = data.get("answers", {}), data.get("notes", {})
    missing = [pid for pid in key if pid not in answers]
    if missing:
        print(f"unanswered: {', '.join(missing)}")
    for name in ("plain", "breath"):
        for voice in ("anitha", "arun", "both"):
            tally = {"geetha": 0, "base": 0, "same": 0}
            for pid, row in key.items():
                if row["section"] != name or pid not in answers:
                    continue
                if voice != "both" and row["voice"] != voice:
                    continue
                pick = answers[pid]
                tally["same" if pick == "same" else row[pick]] += 1
            decided = tally["geetha"] + tally["base"]
            print(f"{name:6s} {voice:6s} geetha {tally['geetha']:2d}  base {tally['base']:2d}  "
                  f"same {tally['same']:2d}   geetha share of decided: {wilson(tally['geetha'], decided)}")
    heard = {"C_tag": 0, "C_notag": 0}
    rows = [(pid, row) for pid, row in key.items() if row["section"] == "tag" and pid in answers]
    for pid, row in rows:
        pick = answers[pid]
        for side in ("A", "B"):
            if pick == side or pick == "both":
                heard[row[side]] += 1
    print(f"tag    arun   breath heard: with tag {heard['C_tag']}/{len(rows)}, "
          f"without tag {heard['C_notag']}/{len(rows)}")
    for pid, note in notes.items():
        if note.strip() and pid in key:
            row = key[pid]
            print(f"note {pid} [{row['section']} {row['voice']} t{row['text']} {row['arm']} "
                  f"A={row['A']} B={row['B']}]: {note.strip()}")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Geetha blind listen</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:Segoe UI,Nirmala UI,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;background:#f6f5f1;color:#1d1d1b}
h1{font-size:22px} h2{margin-top:36px;font-size:18px;border-top:2px solid #1d1d1b;padding-top:14px}
.ask{font-weight:600} .how{background:#fff;border:1px solid #d8d5cc;padding:12px 16px;border-radius:6px;line-height:1.5}
.pair{display:flex;gap:14px;background:#fff;border:1px solid #d8d5cc;border-radius:6px;padding:12px 14px;margin:10px 0}
.pair.done{border-color:#2f7d4f;background:#f3faf5}
.num{font-size:20px;font-weight:700;width:30px;color:#777}
.body{flex:1;min-width:0} .text{font-size:15px;margin-bottom:8px;line-height:1.5}
.players{display:flex;flex-wrap:wrap;align-items:center;gap:8px} .players span{font-weight:700}
audio{height:34px;width:300px;max-width:100%}
.pick{margin-top:8px;display:flex;flex-wrap:wrap;gap:16px} .pick label{cursor:pointer}
.note{margin-top:8px;width:100%;box-sizing:border-box;padding:5px 8px;border:1px solid #ccc;border-radius:4px}
#bar{position:sticky;top:0;background:#1d1d1b;color:#fff;padding:8px 14px;border-radius:0 0 6px 6px;display:flex;justify-content:space-between;align-items:center;z-index:5}
button{font-size:15px;padding:8px 16px;border:0;border-radius:4px;background:#2f7d4f;color:#fff;cursor:pointer}
textarea{width:100%;height:90px;box-sizing:border-box;margin-top:10px}
</style></head><body>
<div id="bar"><span id="count">0 of @@TOTAL@@ answered</span><button onclick="save()">Save answers</button></div>
<h1>Geetha blind listen - 15 Nov</h1>
<div class="how">
Each row has the same sentence spoken twice, A and B. One is one system, one is another; which is which changes from row to row and is hidden.<br>
Use headphones. Play A, play B, pick. Replay as often as you like. If you cannot tell, pick "No real difference" - that is a real answer, not a failure.<br>
All clips are set to the same loudness, so louder never means better here.<br>
Your picks are remembered if you close the page. At the end press <b>Save answers</b> and send the file.
</div>
@@BLOCKS@@
<h2>Finished</h2>
<p><button onclick="save()">Save answers</button> A file <code>blind_answers.json</code> goes to your Downloads folder. The same text appears below in case the download is blocked.</p>
<textarea id="dump" readonly></textarea>
<script>
const STORE = "geetha_blind_2711";
function collect(){
  const answers = {}, notes = {};
  document.querySelectorAll("input[type=radio]:checked").forEach(r => answers[r.name] = r.value);
  document.querySelectorAll("input[data-note]").forEach(n => { if (n.value.trim()) notes[n.dataset.note] = n.value; });
  return {answers, notes};
}
function refresh(){
  const data = collect();
  document.querySelectorAll(".pair").forEach(p => p.classList.toggle("done", p.id in data.answers));
  document.getElementById("count").textContent = Object.keys(data.answers).length + " of @@TOTAL@@ answered";
  document.getElementById("dump").value = JSON.stringify(data);
  try { localStorage.setItem(STORE, JSON.stringify(data)); } catch (e) {}
}
function restore(){
  let data = null;
  try { data = JSON.parse(localStorage.getItem(STORE)); } catch (e) {}
  if (!data) return;
  for (const [pid, value] of Object.entries(data.answers || {})) {
    const r = document.querySelector(`input[name="${pid}"][value="${value}"]`); if (r) r.checked = true;
  }
  for (const [pid, value] of Object.entries(data.notes || {})) {
    const n = document.querySelector(`input[data-note="${pid}"]`); if (n) n.value = value;
  }
}
function save(){
  refresh();
  const blob = new Blob([document.getElementById("dump").value], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "blind_answers.json"; a.click();
}
document.addEventListener("change", refresh); document.addEventListener("input", refresh);
document.addEventListener("play", e => { document.querySelectorAll("audio").forEach(a => { if (a !== e.target) a.pause(); }); }, true);
restore(); refresh();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make")
    scorer = sub.add_parser("score")
    scorer.add_argument("answers", type=Path)
    args = parser.parse_args()
    if args.command == "make":
        make()
    else:
        score(args.answers)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
