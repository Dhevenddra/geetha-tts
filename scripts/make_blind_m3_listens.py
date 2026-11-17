"""Blind listening pages for the two 16 Nov ear tests, and their scoring.

    make breath     12 pairs "old mid breath vs cut-at-the-tag" + 12 pairs "cut-at-the-tag
                    vs no breath", from runs\\m3_mid_split
    make emotion    24 pairs "styled vs plain, same sentence and seed", from
                    runs\\m3_emotion_listen; the page names the feeling to listen for
    score breath|emotion <answers.json>

Both systems in every pair are Geetha in the same voice, so these are blind in full:
only the order is hidden and only the order could give it away.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_blind_ab import PAGE, level, wilson  # noqa: E402

RUNS = Path(r"E:\Geetha\runs")
TESTS = {
    "breath": {"run": RUNS / "m3_mid_split", "report": "m3_mid_split.json", "seed": 1612,
               "title": "Geetha listen 3 - a breath in the middle"},
    "emotion": {"run": RUNS / "m3_emotion_listen", "report": "m3_emotion_listen.json",
                "seed": 1613, "title": "Geetha listen 4 - emotions"},
}
FEELING = {"HAPPY": "happy", "SAD": "sad", "ANGER": "angry", "FEAR": "afraid",
           "SURPRISE": "surprised", "DISGUST": "disgusted"}


def pairs_for(name: str, rows: list[dict]) -> list[dict]:
    """Each pair: question, text, x and y as (label, file), group."""
    out = []
    if name == "breath":
        by = {(r["voice"], r["sentence"], r["arm"]): r for r in rows if r["valid"]}
        for (voice, sentence, arm), row in sorted(by.items()):
            if arm != "split":
                continue
            for other, ask in (("old", "In which one does the breath belong to the sentence?"),
                               ("none", "Which one sounds more like a real person talking?")):
                if (voice, sentence, other) in by:
                    out.append({"group": f"split_vs_{other}", "voice": voice, "ask": ask,
                                "text": row["text"].replace("[breath] ", ""),
                                "x": (other, by[voice, sentence, other]["file"]),
                                "y": ("split", row["file"])})
    else:
        emo = [r for r in rows if r["kind"] == "emo" and r["valid"]]
        styled = {r["file"].replace("_styled.wav", ""): r for r in emo if r["style"] != "CONV"}
        plain = {r["file"].replace("_plain.wav", ""): r for r in emo if r["style"] == "CONV"}
        for key in sorted(set(styled) & set(plain)):
            row = styled[key]
            out.append({"group": row["target"], "voice": row["voice"], "text": row["text"],
                        "ask": f"Which one sounds more {FEELING[row['target']]}?",
                        "x": ("plain", plain[key]["file"]), "y": ("styled", row["file"])})
    return out


def make(name: str) -> None:
    test = TESTS[name]
    rng = random.Random(test["seed"])
    rows = json.loads((test["run"] / test["report"]).read_text(encoding="utf-8"))["rows"]
    invalid = [r["file"] for r in rows if not r["valid"]]
    if invalid:
        print(f"invalid clips, their pairs are left out: {invalid}")
    pairs = pairs_for(name, rows)
    rng.shuffle(pairs)
    flips = [i % 2 == 0 for i in range(len(pairs))]
    rng.shuffle(flips)
    out = test["run"] / "blind"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("p*_[AB].wav"):
        old.unlink()
    key, blocks = {}, [f"<h2>{html.escape(test['title'])}</h2>"
                       "<p class='ask'>Each pair is the same voice saying the same sentence. "
                       "The question is written above each pair.</p>"]
    for number, (pair, flip) in enumerate(zip(pairs, flips), 1):
        pid = f"p{number:02d}"
        a, b = (pair["y"], pair["x"]) if flip else (pair["x"], pair["y"])
        level(test["run"] / "clips" / a[1], out / f"{pid}_A.wav")
        level(test["run"] / "clips" / b[1], out / f"{pid}_B.wav")
        key[pid] = {"group": pair["group"], "voice": pair["voice"], "A": a[0], "B": b[0],
                    "files": {"A": a[1], "B": b[1]}}
        radios = "".join(
            f"<label><input type='radio' name='{pid}' value='{value}'> {label}</label>"
            for value, label in (("A", "A"), ("B", "B"), ("same", "No real difference")))
        blocks.append(
            f"<div class='pair' id='{pid}'><div class='num'>{number}</div>"
            f"<div class='body'><div class='ask'><b>{html.escape(pair['ask'])}</b></div>"
            f"<div class='text'>{html.escape(pair['text'])}</div>"
            f"<div class='players'><span>A</span><audio controls preload='none' src='{pid}_A.wav'></audio>"
            f"<span>B</span><audio controls preload='none' src='{pid}_B.wav'></audio></div>"
            f"<div class='pick'>{radios}</div>"
            f"<input class='note' type='text' data-note='{pid}' placeholder='note (optional)'></div></div>")
    key_path = test["run"] / f"blind_{name}_key.json"
    key_path.write_text(json.dumps({"seed": test["seed"], "pairs": key}, ensure_ascii=False,
                                   indent=1), encoding="utf-8")
    page = (PAGE.replace("@@BLOCKS@@", "\n".join(blocks)).replace("@@TOTAL@@", str(len(pairs)))
            .replace("geetha_blind_2711", f"geetha_blind_{name}_{test['seed']}")
            .replace("blind_answers.json", f"blind_{name}_answers.json")
            .replace("Geetha blind listen - 15 Nov", test["title"]))
    (out / "listen.html").write_text(page, encoding="utf-8")
    print(f"{len(pairs)} pairs -> {out / 'listen.html'}")
    print(f"key (do not open before listening) -> {key_path}")


def score(name: str, answers_path: Path) -> None:
    test = TESTS[name]
    key = json.loads((test["run"] / f"blind_{name}_key.json").read_text(encoding="utf-8"))["pairs"]
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    answers, notes = data.get("answers", {}), data.get("notes", {})
    missing = [pid for pid in key if pid not in answers]
    if missing:
        print(f"unanswered: {', '.join(missing)}")
    wanted = "split" if name == "breath" else "styled"
    groups = sorted({row["group"] for row in key.values()})
    cuts = [("all", None)] if name == "emotion" else []
    cuts += [(g, ("group", g)) for g in groups]
    cuts += [(v, ("voice", v)) for v in ("anitha", "arun")]
    for label, cut in cuts:
        tally = {wanted: 0, "other": 0, "same": 0}
        for pid, row in key.items():
            if pid not in answers or (cut and row[cut[0]] != cut[1]):
                continue
            pick = answers[pid]
            tally["same" if pick == "same" else (wanted if row[pick] == wanted else "other")] += 1
        decided = tally[wanted] + tally["other"]
        print(f"{label:14s} {wanted} {tally[wanted]:2d}  other {tally['other']:2d}  "
              f"same {tally['same']:2d}   {wanted} share of decided: "
              f"{wilson(tally[wanted], decided)}")
    for pid, note in notes.items():
        if note.strip() and pid in key:
            row = key[pid]
            print(f"note {pid} [{row['group']} {row['voice']} A={row['A']} B={row['B']}]: "
                  f"{note.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    maker = sub.add_parser("make")
    maker.add_argument("test", choices=sorted(TESTS))
    scorer = sub.add_parser("score")
    scorer.add_argument("test", choices=sorted(TESTS))
    scorer.add_argument("answers", type=Path)
    args = parser.parse_args()
    if args.command == "make":
        make(args.test)
    else:
        score(args.test, args.answers)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
