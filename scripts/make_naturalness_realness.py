r"""Build Naturalness-Ta REALNESS: the 40 real-vs-rendered pairs (runbook §6).

    py -3.11 scripts\make_naturalness_realness.py [--dry-run]

This is the project's **headline** metric: a listener hears the real Rasa recording and
Geetha's rendering of the same sentence in the same voice, in random order, and picks
the real one. 5 listeners x 40 = 200 judgements; 50% means indistinguishable.

Why this script exists at all
-----------------------------
It was missing. ``make_naturalness_ta.py`` builds the CAPABILITY set and says so in its
own docstring -- "REALNESS: 40 sentences drawn from Rasa test... Selection, not
authoring" -- and that selection was never run. The B4 runlog records only
``event 36 / filler 24 / clean 24``, which is capability. So the eval the release leads
with had no prompt set, while STATUS has been (correctly) reporting the capability set
by name. Found on Day 14 while a pod hunt was running.

What makes a usable realness pair
---------------------------------
* **The sentence must be in Rasa *test*.** All 3,601 test sentences are safe as prompts:
  the 820 that also appear in train were removed from *training* by step 2 (leak table
  = 0), so the model has not been trained on any of them. Verified here rather than
  assumed.
* **Balanced across styles and both voices**, because a pick rate averaged over a set
  that is 90% one emotion measures that emotion, not naturalness.
* **The real recording has to be clean**, since it is half of every pair. The same QC
  gates that filter training audio apply: no clipping, no silence, sane edge silence.
  A listener who can pick the real one because ours is the quieter file has told us
  nothing about naturalness.
* **1.5-10 s.** Too short and there is nothing to judge; too long and 200 judgements
  becomes an hour of listening per person.

Output: ``eval_vault\naturalness_ta_v1\realness_ta_v1.jsonl`` plus the extracted real
audio under ``realness_audio\``, so the pairs can be served without touching ``raw\``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geetha.audio.qc import clip_qc  # noqa: E402
from geetha.data.holdout import HoldoutIndex  # noqa: E402
from geetha.data.manifest import qc_verdict  # noqa: E402
from geetha.text.normalise import norm_text  # noqa: E402

ROOT = Path(r"E:\Geetha")
RASA = ROOT / "raw" / "hf" / "rasa" / "Tamil"
TEXT_NORM = ROOT / "processed" / "text_norm" / "rasa"
OUT_DIR = ROOT / "eval_vault" / "naturalness_ta_v1"
OUT = OUT_DIR / "realness_ta_v1.jsonl"
AUDIO_DIR = OUT_DIR / "realness_audio"

TARGET = 40
SEED = 20261104
MIN_SECONDS, MAX_SECONDS = 1.5, 10.0
# CONV plus the six trained emotions: the styles the release actually claims.
WANTED_STYLES = ("CONV", "HAPPY", "SAD", "ANGER", "FEAR", "SURPRISE", "DISGUST")
VOICES = {"TAM_F": "anitha", "TAM_M": "arun"}


def log(message: str) -> None:
    print(message, flush=True)


def training_texts() -> HoldoutIndex:
    """Every sentence that survived into training, as a leak index to check against."""
    entries: list[tuple[str, str]] = []
    for path in sorted(TEXT_NORM.glob("*.parquet")):
        table = pq.read_table(path, columns=["text", "keep"])
        for text, keep in zip(table["text"].to_pylist(), table["keep"].to_pylist()):
            if keep and text:
                entries.append(("rasa_train_kept", text))
    return HoldoutIndex(entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log("indexing what actually survived into Rasa training")
    trained = training_texts()
    log(f"  {len(trained):,} distinct trained sentences")

    shards = sorted(RASA.glob("test-*.parquet"))
    if not shards:
        log(f"!! no Rasa test shards under {RASA}")
        return 1

    candidates: list[dict] = []
    rejected: Counter[str] = Counter()
    for path in shards:
        table = pq.read_table(path, columns=["filename", "text", "style", "gender", "audio"])
        rows = table.to_pylist()
        for index, row in enumerate(rows):
            style = (row["style"] or "").strip().upper()
            if style not in WANTED_STYLES:
                rejected["style_not_claimed"] += 1
                continue
            text = row["text"] or ""
            if not norm_text(text):
                rejected["empty_text"] += 1
                continue
            # The leak direction that matters: a test sentence the model was trained on
            # would be heard fluently and the pair would measure memorisation.
            if trained.source_of(text) is not None:
                rejected["in_training"] += 1
                continue
            voice = VOICES.get("_".join(str(row["filename"]).split("_")[:2]))
            if voice is None:
                rejected["unknown_voice"] += 1
                continue

            blob = row["audio"]
            wave, rate = sf.read(io.BytesIO(blob["bytes"]), dtype="float32", always_2d=True)
            measured = clip_qc(wave, rate)
            qc_row = {**measured.as_row(), "error": "", "sample_rate": rate}
            verdict = qc_verdict(qc_row)
            if verdict:
                rejected[f"qc_{verdict}"] += 1
                continue
            if not (MIN_SECONDS <= measured.duration_s <= MAX_SECONDS):
                rejected["duration_band"] += 1
                continue

            candidates.append({
                "shard": path.name, "row_index": index,
                "filename": row["filename"], "text": text, "style": style,
                "voice": voice, "duration_s": round(measured.duration_s, 3),
                "rms_dbfs": round(measured.rms_dbfs, 2), "sample_rate": rate,
                "wave": wave, "clip_path": blob["path"],
            })

    log(f"\n{len(candidates):,} usable candidates; rejected:")
    for reason, count in rejected.most_common():
        log(f"    {reason:24s} {count:>6,}")
    if len(candidates) < TARGET:
        log(f"!! only {len(candidates)} candidates for {TARGET} pairs")
        return 1

    # Stratify by (style, voice) so the pick rate is not dominated by one cell. Round
    # robin over cells rather than a flat sample, which on Rasa's style distribution
    # would come back mostly CONV and WIKI.
    rng = random.Random(SEED)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in candidates:
        cells[(row["style"], row["voice"])].append(row)
    for bucket in cells.values():
        rng.shuffle(bucket)

    order = sorted(cells)
    chosen: list[dict] = []
    while len(chosen) < TARGET:
        progressed = False
        for key in order:
            if cells[key] and len(chosen) < TARGET:
                chosen.append(cells[key].pop())
                progressed = True
        if not progressed:
            break
    if len(chosen) < TARGET:
        log(f"!! stratification only filled {len(chosen)} of {TARGET}")
        return 1

    log(f"\nselected {len(chosen)} pairs")
    for label, counts in (("style", Counter(r["style"] for r in chosen)),
                          ("voice", Counter(r["voice"] for r in chosen))):
        log(f"  by {label}: {dict(sorted(counts.items()))}")
    seconds = sum(r["duration_s"] for r in chosen)
    log(f"  total real audio {seconds:.1f} s; a listener hears both halves of 40 pairs "
        f"~= {2 * seconds / 60:.1f} min of audio")

    if args.dry_run:
        log("\ndry run: nothing written")
        return 0

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(chosen):
        item_id = f"r{position:02d}"
        name = f"{item_id}_{row['voice']}_{row['style']}.wav"
        mono = row["wave"].mean(axis=1) if row["wave"].ndim == 2 else row["wave"]
        sf.write(AUDIO_DIR / name, mono.astype(np.float32), row["sample_rate"])
        digest = hashlib.sha256((AUDIO_DIR / name).read_bytes()).hexdigest()
        records.append({
            "id": item_id,
            "text": row["text"],
            "style": row["style"],
            "voice": row["voice"],
            "real_audio": f"realness_audio/{name}",
            "real_audio_sha256": digest,
            "duration_s": row["duration_s"],
            "rms_dbfs": row["rms_dbfs"],
            "sample_rate": row["sample_rate"],
            "source": {"corpus": "rasa", "split": "test",
                       "shard": row["shard"], "row_index": row["row_index"],
                       "filename": row["filename"]},
            # Filled by the eval run; recorded here so the schema is fixed before any
            # audio exists and cannot be shaped around a result.
            "rendered_audio": None,
        })

    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"\nwrote {len(records)} records -> {OUT}")
    log(f"wrote {len(records)} real recordings -> {AUDIO_DIR}")
    log("\nNOT yet frozen: add both to manifests\\eval_freeze.csv before any number is "
        "measured on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
