"""A page to pick the take of each showcase sentence that goes on the public pages.

    make                 runs\\m3_showcase\\pick\\listen.html, one block per sentence
    collect <answers>    copies the chosen takes to results\\release_samples\\ with a
                         samples.json (text, voice, style, seed, "selected from N takes")

Not a blind test: the listener is choosing, and the samples are labelled as selected.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_blind_ab import PAGE, level  # noqa: E402

RUN = Path(r"E:\Geetha\runs\m3_showcase")
OUT = RUN / "pick"
SAMPLES = Path(r"E:\Geetha\results\release_samples")


def groups() -> dict[str, list[dict]]:
    report = json.loads((RUN / "m3_showcase.json").read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {}
    for row in report["rows"]:
        if row["valid"]:
            out.setdefault(row["name"], []).append(dict(row, seed=report["seed"] + 10 * row["seed_index"]))
    return out


def make() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = ["<h2>Pick the take to show</h2><p class='ask'>For each sentence choose the take "
              "you would be happy to put your name on, or 'none of these'.</p>"]
    everything = groups()
    for number, (name, takes) in enumerate(everything.items(), 1):
        players, radios = "", ""
        for take in takes:
            label = take["file"].rsplit("_", 1)[1].removesuffix(".wav")
            level(RUN / "clips" / take["file"], OUT / take["file"])
            players += f"<span>{label}</span><audio controls preload='none' src='{take['file']}'></audio>"
            radios += f"<label><input type='radio' name='{name}' value='{take['file']}'> {label}</label>"
        radios += f"<label><input type='radio' name='{name}' value='none'> none of these</label>"
        first = takes[0]
        blocks.append(
            f"<div class='pair' id='{name}'><div class='num'>{number}</div><div class='body'>"
            f"<div class='ask'><b>{first['voice']} &middot; {first['style']}</b></div>"
            f"<div class='text'>{html.escape(first['text'])}</div>"
            f"<div class='players'>{players}</div><div class='pick'>{radios}</div>"
            f"<input class='note' type='text' data-note='{name}' placeholder='note (optional)'></div></div>")
    page = (PAGE.replace("@@BLOCKS@@", "\n".join(blocks)).replace("@@TOTAL@@", str(len(everything)))
            .replace("geetha_blind_2711", "geetha_showcase_pick")
            .replace("blind_answers.json", "showcase_picks.json")
            .replace("Geetha blind listen - 15 Nov", "Geetha - pick the showcase takes"))
    (OUT / "listen.html").write_text(page, encoding="utf-8")
    print(f"{len(everything)} sentences -> {OUT / 'listen.html'}")


def collect(answers_path: Path) -> None:
    data = json.loads(answers_path.read_text(encoding="utf-8"))
    everything = groups()
    if SAMPLES.exists():
        shutil.rmtree(SAMPLES)
    SAMPLES.mkdir(parents=True)
    listing = []
    for name, takes in everything.items():
        pick = data.get("answers", {}).get(name)
        chosen = next((t for t in takes if t["file"] == pick), None)
        if not chosen:
            print(f"no take chosen for {name} ({pick})")
            continue
        shutil.copy2(OUT / chosen["file"], SAMPLES / f"{name}.wav")
        listing.append({"file": f"{name}.wav", "voice": chosen["voice"], "style": chosen["style"],
                        "text": chosen["text"], "seed": chosen["seed"],
                        "selected_from_takes": len(takes)})
    (SAMPLES / "samples.json").write_text(json.dumps(listing, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"{len(listing)} samples -> {SAMPLES}")
    for pid, note in data.get("notes", {}).items():
        if note.strip():
            print(f"note {pid}: {note.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make")
    sub.add_parser("collect").add_argument("answers", type=Path)
    args = parser.parse_args()
    make() if args.command == "make" else collect(args.answers)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
