"""Blind A/B listening sheet: Geetha against the two open Tamil rivals, from benchmark v2.

    make   picks prompts where both systems made a valid clip (4 per source per rival,
           voices balanced), level-matches them, writes blind\\listen.html and keeps the
           key outside the folder the listener opens.
    score  reads the answers file the page downloads and prints the tallies.

What this can and cannot show: the rivals speak in other voices, and the listener knows
anitha and arun by now, so the test is blind to ORDER but not truly to SYSTEM. It is one
listener's preference on a few sentences, reported as that.

Usage:
    python scripts\\make_blind_rivals.py make
    python scripts\\make_blind_rivals.py score E:\\path\\to\\blind_rivals_answers.json
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
BENCH = Path(r"E:\Geetha\work\bench_ta_v2\bench_ta_v2.json")
OURS, RIVALS = RUNS / "m3_bench_ours", RUNS / "m3_bench_rivals"
OUT = RIVALS / "blind"
KEY = RIVALS / "blind_rivals_key.json"
SEED = 1611
PER_SOURCE = 4
RIVAL_NAMES = ("rumik_oss_1", "snortts_indic_v0")
OPTIONS = [("A", "A"), ("B", "B"), ("same", "No real difference")]
MIN_SECONDS = 2.0       # very short clips give the ear nothing to compare


def load(folder: Path, system: str) -> dict:
    return json.loads((folder / f"bench_{system}.json").read_text(encoding="utf-8"))["clips"]


def make() -> None:
    rng = random.Random(SEED)
    prompts = json.loads(BENCH.read_text(encoding="utf-8"))["prompts"]
    geetha = load(OURS, "geetha")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("p*_[AB].wav"):
        old.unlink()
    used: set[str] = set()
    pairs = []
    for rival in RIVAL_NAMES:
        theirs = load(RIVALS, rival)
        for source in ("fleurs", "dir", "cap"):
            pool = [p for p in prompts if p["source"] == source and p["id"] not in used
                    and geetha.get(p["id"], {}).get("valid") and theirs.get(p["id"], {}).get("valid")
                    and min(geetha[p["id"]]["seconds"], theirs[p["id"]]["seconds"]) >= MIN_SECONDS]
            rng.shuffle(pool)
            chosen = []
            for voice in ("anitha", "arun") * PER_SOURCE:        # alternate the two voices
                pick = next((p for p in pool if p["voice"] == voice and p not in chosen), None)
                if pick and len(chosen) < PER_SOURCE:
                    chosen.append(pick)
            if len(chosen) < PER_SOURCE:
                print(f"only {len(chosen)} usable prompts for {rival} / {source}")
            for p in chosen:
                used.add(p["id"])
                pairs.append({"prompt": p, "rival": rival,
                              "x": (rival, RIVALS / "clips" / theirs[p["id"]]["file"]),
                              "y": ("geetha", OURS / "clips" / geetha[p["id"]]["file"])})
    rng.shuffle(pairs)
    flips = [i % 2 == 0 for i in range(len(pairs))]
    rng.shuffle(flips)
    key, blocks = {}, ["<h2>Geetha and two other Tamil systems</h2>"
                       "<p class='ask'>Which one sounds more like a real person talking?</p>"]
    for number, (pair, flip) in enumerate(zip(pairs, flips), 1):
        pid = f"p{number:02d}"
        a, b = (pair["y"], pair["x"]) if flip else (pair["x"], pair["y"])
        level(a[1], OUT / f"{pid}_A.wav")
        level(b[1], OUT / f"{pid}_B.wav")
        p = pair["prompt"]
        key[pid] = {"prompt": p["id"], "source": p["source"], "voice": p["voice"],
                    "rival": pair["rival"], "A": a[0], "B": b[0]}
        radios = "".join(
            f"<label><input type='radio' name='{pid}' value='{value}'> {html.escape(label)}</label>"
            for value, label in OPTIONS)
        blocks.append(
            f"<div class='pair' id='{pid}'><div class='num'>{number}</div>"
            f"<div class='body'><div class='text'>{html.escape(p['text'])}</div>"
            f"<div class='players'><span>A</span><audio controls preload='none' src='{pid}_A.wav'></audio>"
            f"<span>B</span><audio controls preload='none' src='{pid}_B.wav'></audio></div>"
            f"<div class='pick'>{radios}</div>"
            f"<input class='note' type='text' data-note='{pid}' placeholder='note (optional)'></div></div>")
    KEY.write_text(json.dumps({"seed": SEED, "pairs": key}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    page = (PAGE.replace("@@BLOCKS@@", "\n".join(blocks)).replace("@@TOTAL@@", str(len(pairs)))
            .replace("geetha_blind_2711", f"geetha_blind_rivals_{SEED}")
            .replace("blind_answers.json", "blind_rivals_answers.json")
            .replace("Geetha blind listen - 15 Nov", "Geetha blind listen 2 - other Tamil systems"))
    (OUT / "listen.html").write_text(page, encoding="utf-8")
    print(f"{len(pairs)} pairs -> {OUT / 'listen.html'}")
    print(f"key (do not open before listening) -> {KEY}")


def score(answers_path: Path) -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))["pairs"]
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    answers, notes = data.get("answers", {}), data.get("notes", {})
    missing = [pid for pid in key if pid not in answers]
    if missing:
        print(f"unanswered: {', '.join(missing)}")
    for rival in RIVAL_NAMES:
        for group in ("all", "fleurs", "dir", "cap"):
            tally = {"geetha": 0, "rival": 0, "same": 0}
            for pid, row in key.items():
                if row["rival"] != rival or pid not in answers:
                    continue
                if group != "all" and row["source"] != group:
                    continue
                pick = answers[pid]
                tally["same" if pick == "same" else
                      ("geetha" if row[pick] == "geetha" else "rival")] += 1
            decided = tally["geetha"] + tally["rival"]
            print(f"{rival:17s} {group:6s} geetha {tally['geetha']:2d}  rival {tally['rival']:2d}  "
                  f"same {tally['same']:2d}   geetha share of decided: "
                  f"{wilson(tally['geetha'], decided)}")
    for pid, note in notes.items():
        if note.strip() and pid in key:
            row = key[pid]
            print(f"note {pid} [{row['rival']} {row['source']} {row['prompt']} {row['voice']} "
                  f"A={row['A']} B={row['B']}]: {note.strip()}")


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
