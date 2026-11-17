r"""Turn the breath generator's clips into Stage C training rows (D-057).

    .venv\Scripts\python.exe scripts\build_breath_rows.py E:\Geetha\runs\m2_breath_gen

A clip becomes a row only if all of this holds, each measured here:

  * the kernel's validity gate passed it;
  * every inserted span lies in a gap between two words (forced alignment), and none
    trails the last word;
  * at least one inserted span holds a pause the D-030 instrument scores as a breath --
    and the tag is written ONLY at spans that do, so the tag never labels silence;
  * the voice is still the voice (WeSpeaker similarity to the same speaker's P1 clips);
  * the words are still there (mean alignment score, a garble guard).

The tag goes before the word the breath precedes; the generator's own prompt tags are
discarded. One breath position per clip is the operator's rule ("never forced"): mid
clips keep at most MAX_TAGS tags.

Writes ``rows.json`` (every clip, kept or not, with the reason) into the run folder and
rewrites the SELECTION block of ``kaggle\m3_breath_stagec\m3_breath_stagec.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geetha.audio.align import Aligner, boundary_for_span, place_tags  # noqa: E402

TAG = "<|nv_breath|>"
KERNEL = REPO / "kaggle" / "m3_breath_stagec" / "m3_breath_stagec.py"
MIN_COVERED = 0.8          # share of the inserted span inside the word gap
MIN_WORD_SCORE = 0.45      # mean forced-alignment score over the words
MAX_TAGS = 2


def speaker_embedder():
    sys.path.insert(0, str(REPO / "scripts"))
    import speaker_similarity as ss  # type: ignore

    return ss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--min-similarity", type=float, default=None,
                        help="default: the lowest P1-to-P1 similarity of that voice minus 0.08")
    parser.add_argument("--no-kernel", action="store_true", help="do not rewrite SELECTION")
    args = parser.parse_args()

    clips = args.run / "clips"
    scored = {r["file"]: r for r in json.loads(
        (args.run / "m2_breath_gen.json").read_text(encoding="utf-8"))["rows"]}
    records = [json.loads(line) for line in
               (args.run / "tokens.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ss = speaker_embedder()

    def vector(file: str) -> np.ndarray:
        raw = np.asarray(ss.embed(clips / file), dtype=np.float64).reshape(-1)
        return raw / np.linalg.norm(raw)

    reference: dict[str, list[np.ndarray]] = {}
    for record in records:
        if record["arm"] == "P1" and record["valid"]:
            reference.setdefault(record["voice"], []).append(vector(record["file"]))
    floor: dict[str, float] = {}
    centre: dict[str, np.ndarray] = {}
    for voice, vectors in reference.items():
        stack = np.stack(vectors)
        centre[voice] = stack.mean(axis=0) / np.linalg.norm(stack.mean(axis=0))
        own = [float(v @ centre[voice]) for v in stack]
        floor[voice] = (args.min_similarity if args.min_similarity is not None
                        else min(own) - 0.08)
        print(f"{voice}: {len(vectors)} reference clips, own similarity "
              f"{min(own):.3f}-{max(own):.3f}, keep >= {floor[voice]:.3f}")

    aligner = Aligner("cpu")
    out_rows, selection = [], {}
    for record in records:
        if record["arm"] not in ("lead", "mid"):
            continue
        row = {"file": record["file"], "voice": record["voice"], "arm": record["arm"],
               "kept": False, "reason": ""}
        out_rows.append(row)
        if not record["valid"]:
            row["reason"] = "invalid clip"
            continue
        spans = record["donor_spans"]
        if not spans:
            row["reason"] = "nothing was inserted"
            continue
        words = " ".join(record["prompt_text"].replace(TAG, " ").split()).split()
        wave, rate = sf.read(clips / record["file"], dtype="float32")
        try:
            aligned = aligner.words(wave, rate, words)
        except Exception as problem:  # noqa: BLE001
            row["reason"] = f"alignment failed: {problem}"
            continue
        row["word_score"] = round(float(np.mean([w.score for w in aligned])), 3)
        if row["word_score"] < MIN_WORD_SCORE:
            row["reason"] = "words poorly aligned (garbled?)"
            continue
        breaths = [p for p in scored[record["file"]].get("pauses", []) if p["breath"]]
        tags, problem = [], ""
        for span in spans:
            placement = boundary_for_span(aligned, span["from_s"], span["to_s"], len(wave) / rate)
            if placement.covered < MIN_COVERED:
                problem = "an inserted span overlaps a word"
                break
            if placement.before_word >= len(words):
                problem = "an inserted span trails the last word"
                break
            heard = any(span["from_s"] - 0.1 <= p["at_s"] + p["dur_s"] / 2 <= span["to_s"] + 0.1
                        for p in breaths)
            if heard:
                tags.append(placement.before_word)
        if problem:
            row["reason"] = problem
            continue
        tags = sorted(set(tags))
        if not tags:
            row["reason"] = "no inserted span scores as a breath"
            continue
        if len(tags) > MAX_TAGS:
            row["reason"] = "more breaths than the rule allows"
            continue
        row["similarity"] = round(float(vector(record["file"]) @ centre[record["voice"]]), 3)
        if row["similarity"] < floor[record["voice"]]:
            row["reason"] = "voice drifted"
            continue
        row.update(kept=True, tags_before_words=tags, text=place_tags(words, tags, TAG),
                   spans=len(spans))
        selection[record["file"]] = row["text"]

    (args.run / "rows.json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    reasons: dict[str, int] = {}
    for row in out_rows:
        key = "KEPT" if row["kept"] else row["reason"]
        reasons[f"{row['voice']} {row['arm']}: {key}"] = \
            reasons.get(f"{row['voice']} {row['arm']}: {key}", 0) + 1
    for key in sorted(reasons):
        print(f"  {reasons[key]:4d}  {key}")
    print(f"kept {len(selection)} of {len(out_rows)}")

    if not args.no_kernel:
        source = KERNEL.read_text(encoding="utf-8")
        begin = source.index("# SELECTION-BEGIN")
        end = source.index("# SELECTION-END")
        block = ("# SELECTION-BEGIN\nSELECTION: dict[str, str] = "
                 + json.dumps(selection, ensure_ascii=False, indent=1) + "\n")
        KERNEL.write_text(source[:begin] + block + source[end:], encoding="utf-8", newline="\n")
        print(f"wrote SELECTION ({len(selection)} clips) into {KERNEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
