"""Recover the breath library from the generator run and write release\\breaths.json.

The generator (kaggle\\m2_breath_gen) cut 40 breaths out of the model's own ivr-voice
speech and wrote them into anitha/arun clips as codec tokens. It saved every clip's
tokens and where the breath sits, but not the library itself. A breath is therefore read
back from the clips that carry it, and accepted only if every clip carrying it holds the
same tokens.

Each breath is then ranked by what happened to the clips it was put in
(scripts\\build_breath_rows.py): inside a gap between words, scored as a breath by the
D-030 instrument, speaker kept. The generator tried each breath in ONE voice only, so a
breath is shipped for the voice it passed in and never for the other.

Usage:  python scripts\\build_release_breaths.py [run_dir]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

FRAME_TOKENS = 7
MIN_USES = 6          # clips a breath was tried in
MIN_KEPT = 0.75       # share of those clips that passed every filter
OUT = Path(__file__).resolve().parents[1] / "release" / "breaths.json"


def main() -> int:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:\Geetha\runs\m2_breath_gen")
    library = json.loads((run / "breath_library.json").read_text(encoding="utf-8"))
    kept = {row["file"]: row for row in json.loads((run / "rows.json").read_text(encoding="utf-8"))}
    seen: dict[int, set] = collections.defaultdict(set)
    uses: dict[int, list] = collections.defaultdict(list)
    with open(run / "tokens.jsonl", encoding="utf-8") as handle:
        for line in handle:
            clip = json.loads(line)
            index = clip["library_first"]
            if index < 0 or len(clip["donor_spans"]) != 1 or clip["file"] not in kept:
                continue
            span = clip["donor_spans"][0]
            if "from_frame" in span:
                first, stop = span["from_frame"], span["to_frame"]
            elif span["from_s"] == 0:
                first, stop = 0, span["frames"]
            else:
                continue
            if stop - first != library[index]["frames"]:
                continue
            seen[index].add(tuple(clip["tokens"][first * FRAME_TOKENS:stop * FRAME_TOKENS]))
            uses[index].append((clip["voice"], clip["arm"], bool(kept[clip["file"]]["kept"])))

    shipped, report = [], []
    for index in sorted(uses):
        if len(seen[index]) != 1:
            report.append(f"breath {index:2d}: clips disagree on its tokens ({len(seen[index])} versions) - dropped")
            continue
        rows = uses[index]
        rate = sum(ok for _, _, ok in rows) / len(rows)
        per_voice = {voice: [ok for v, _, ok in rows if v == voice] for voice in ("anitha", "arun")}
        voices = [v for v in per_voice if per_voice[v]]
        ship = len(rows) >= MIN_USES and rate >= MIN_KEPT and len(voices) == 1
        report.append(f"breath {index:2d}: used {len(rows):2d}, kept {100 * rate:3.0f}%  "
                      f"anitha {sum(per_voice['anitha'])}/{len(per_voice['anitha'])}  "
                      f"arun {sum(per_voice['arun'])}/{len(per_voice['arun'])}  "
                      f"score {library[index]['score']:.2f}  {library[index]['dur_s']:.2f} s"
                      f"{'  SHIP' if ship else ''}")
        if ship:
            tokens = list(next(iter(seen[index])))
            shipped.append({"id": index, "voice": voices[0], "seconds": library[index]["dur_s"],
                            "detector_score": library[index]["score"], "kept_rate": round(rate, 3),
                            "tokens": tokens})
    print("\n".join(report))
    shipped.sort(key=lambda b: (b["voice"], -b["kept_rate"], -b["detector_score"]))
    OUT.write_text(json.dumps({
        "what": "Breaths the model itself produced, as codec tokens, written into the token "
                "stream by inference.py when the text carries a breath tag.",
        "frame_tokens": FRAME_TOKENS, "breaths": shipped}, indent=1), encoding="utf-8")
    count = collections.Counter(b["voice"] for b in shipped)
    print(f"\n{len(shipped)} of {len(library)} breaths shipped {dict(count)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
