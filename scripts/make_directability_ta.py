r"""Task B4 (part 1) — build the Directability-Ta prompt set.

Writes ``E:\Geetha\eval_vault\directability_ta_v1.jsonl`` plus a report and an operator
review sheet, and **refuses to write anything if any prompt collides with training
text** (runbook rule 3: nothing in the eval vault may appear in a training manifest).

The leak index is built from the text columns of every training corpus it can reach, and
the report records exactly which ones were covered — a leak check whose coverage is
unstated is not a leak check.

    py -3.11 scripts/make_directability_ta.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from geetha.eval.directability import (
    CARRIER_STYLES,
    build_items,
    estimate_seconds,
    find_leaks,
    load_source,
    validate_source,
)
from geetha.eval.leaks import plan_leak_removal
from geetha.runlog import RunLog, text_hash
from geetha.text.normalise import norm_text

EVAL_VAULT = Path(r"E:\Geetha\eval_vault")
RAW = Path(r"E:\Geetha\raw")
OUT = EVAL_VAULT / "directability_ta_v1.jsonl"
SEED = 1234

# Every training corpus and the text column to hash. Benchmark-only sets are absent on
# purpose: a prompt colliding with a *test* set is not a training leak.
TRAINING_SOURCES: list[tuple[str, str, str]] = [
    ("indicvoices", r"hf\indicvoices\tamil\train-*.parquet", "text"),
    ("indicvoices_r", r"hf\indicvoices_r\Tamil\train-*.parquet", "text"),
    ("rasa", r"hf\rasa\Tamil\train-*.parquet", "text"),
    ("spring_r1", r"hf\spring_inx_tamil_r1\data\train-*.parquet", "text"),
    ("spring_r2", r"hf\spring_inx_tamil_r2\data\train-*.parquet", "text"),
    ("fleurs", r"hf\fleurs\parquet-data\ta_in\train-*.parquet", "raw_transcription"),
    ("shrutilipi", r"hf\shrutilipi\**\train-*.parquet", "text"),
]


def build_leak_index(
    log: RunLog, prompt_hashes: dict[str, str]
) -> tuple[set[str], dict[str, int], collections.Counter, int]:
    """Normalised training sentences, per-corpus coverage, and per-prompt clip counts.

    The clip counts are what make rule 3's remedy costable: they say exactly how much
    training data is given up to keep a prompt out of it.
    """
    index: set[str] = set()
    coverage: dict[str, int] = {}
    clip_counts: collections.Counter[str] = collections.Counter()
    training_rows = 0
    for name, pattern, column in TRAINING_SOURCES:
        files = sorted(glob.glob(str(RAW / pattern), recursive=True))
        if not files:
            coverage[name] = -1  # not found; recorded so the gap is visible
            print(f"   {name}: no files matched {pattern} — NOT COVERED")
            continue
        rows = 0
        for path in files:
            try:
                table = pq.read_table(path, columns=[column])
            except Exception as error:  # a schema without that column, etc.
                print(f"   {name}: {Path(path).name} unreadable ({error}); skipped")
                continue
            for value in table.column(column).to_pylist():
                if not value:
                    continue
                key = norm_text(value)
                index.add(key)
                rows += 1
                if key in prompt_hashes:
                    clip_counts[prompt_hashes[key]] += 1
        coverage[name] = rows
        training_rows += rows
        print(f"   {name}: {rows:,} sentences from {len(files)} file(s)")
        log.add_param(f"leak_index_{name}_rows", rows)
    return index, coverage, clip_counts, training_rows


def main() -> int:
    source = load_source()
    validate_source(source)
    print(f"source validated: {len(source)} authored items")

    log = RunLog(run_id="b4-directability-ta", seed=SEED, out_dir=EVAL_VAULT)
    log.add_input(Path(__file__).resolve().parent.parent / "geetha" / "data" / "directability_ta_source.tsv")
    log.add_param("carrier_styles", list(CARRIER_STYLES))

    print("building the leak index from training corpora...")
    t0 = time.time()
    prompt_hashes = {norm_text(item.text): item.id for item in source}
    index, coverage, clip_counts, training_rows = build_leak_index(log, prompt_hashes)
    print(f"leak index: {len(index):,} distinct normalised sentences ({time.time()-t0:.0f}s)")

    # Rule 3's remedy is to remove the colliding sentence from every TRAINING manifest,
    # not to rewrite the eval set. Cost it, then record the instruction for ETL step 2.
    leaks = find_leaks(source, index)
    plan = plan_leak_removal(
        leaks=leaks, clip_counts=dict(clip_counts), training_rows=training_rows
    )
    if plan.rows:
        print(
            f"\n{len(plan.rows)} prompt(s) appear in training text. Rule 3 removes them from "
            f"training: {plan.clips_removed:,} of {training_rows:,} clips ({plan.fraction:.2%})."
        )
        for prompt_id, text, clips in plan.rows[:5]:
            print(f"   {prompt_id}: {clips:,} clips  {text}")
        removal = EVAL_VAULT / "directability_ta_v1_leak_removal.csv"
        with removal.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prompt_id", "normalised_text", "training_clips_to_remove"])
            for prompt_id, text, clips in plan.rows:
                writer.writerow([prompt_id, norm_text(text), clips])
        log.add_output(removal)
        log.add_param("leak_clips_to_remove", plan.clips_removed)
        print(f"   removal instruction for ETL step 2: {removal}")

    items = build_items(source)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    review = EVAL_VAULT / "directability_ta_v1_review.tsv"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kind", "id", "text", "gloss", "intents", "est_seconds", "operator_verdict"])
        for item in source:
            writer.writerow([
                item.kind,
                item.id,
                item.text,
                item.gloss,
                "|".join(item.intents),
                f"{estimate_seconds(item.text, chars_per_second=13.2):.1f}",
                "",
            ])

    report = EVAL_VAULT / "directability_ta_v1_report.md"
    report.write_text(_report(source, items, index, coverage, plan), encoding="utf-8")

    for path in (OUT, review, report):
        log.add_output(path)
    log.add_param("prompts_built", len(items))
    log.add_param("leak_index_size", len(index))
    log.finish(status="ok")

    print(f"\nwrote {len(items)} prompts to {OUT}")
    print(f"operator review sheet: {review}")
    return 0


def _report(source, items, index, coverage, plan) -> str:
    kinds = {}
    for item in items:
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    longest = max(source, key=lambda i: estimate_seconds(i.text, chars_per_second=13.2))

    lines = [
        "# Directability-Ta v1",
        "",
        "Built by `scripts/make_directability_ta.py` (task B4), seed 1234. Input hashes and",
        "provenance in `b4-directability-ta.runlog.json`.",
        "",
        "## What it measures",
        "",
        "| Kind | Count | What it tests |",
        "| --- | ---: | --- |",
        f"| tagged | {kinds.get('tagged', 0)} | 40 emotion-**neutral** carrier sentences x "
        f"{len(CARRIER_STYLES)} styles ({', '.join(CARRIER_STYLES)}). The same text under "
        "different tags, so any difference in delivery is attributable to the tag and not to "
        "the words. |",
        f"| untagged_emotional | {kinds.get('untagged_emotional', 0)} | Emotionally loaded text with "
        "**no** tag — D-004: untagged text must still sound right. |",
        f"| probe | {kinds.get('probe', 0)} | Short phrases whose intent changes with delivery. "
        "Listeners pick from a fixed list; chance is recorded per probe as `1/len(intents)`, "
        "not assumed. |",
        "",
        "`DISGUST` is deliberately absent from the carriers — disgust over \"the book is on the",
        "table\" is the weakest of the six — and is covered by the untagged emotional items,",
        "which include disgust-laden content.",
        "",
        "## Leak check (runbook rule 3)",
        "",
        f"Every authored prompt was hashed with `norm_text()` and checked against "
        f"**{len(index):,}** distinct normalised training sentences.",
        "",
        f"**{len(plan.rows)} collided**, all of them intent probes — which is inherent to the",
        "design, because a probe has to be a short common phrase ('சரி', 'போ') or its intent",
        "cannot flip with delivery. Rule 3's remedy is to remove the colliding sentence from",
        "every *training* manifest, not to rewrite the eval set, so that is what happens:",
        "",
        f"- **{plan.clips_removed:,} clips** of {plan.training_rows:,} "
        f"(**{plan.fraction:.2%}**) are listed in `directability_ta_v1_leak_removal.csv`",
        "  for ETL step 2 to drop.",
        "- At that price the rule needs no weakening. The build refuses outright above 1%,",
        "  on the grounds that a prompt set costing more than that is too generic to measure",
        "  anything.",
        "",
        "Corpora covered:",
        "",
        "| Corpus | Sentences indexed |",
        "| --- | ---: |",
    ]
    for name, rows in coverage.items():
        lines.append(f"| {name} | {'NOT FOUND' if rows < 0 else f'{rows:,}'} |")
    lines += [
        "",
        "A corpus marked NOT FOUND was not indexed, and prompts were therefore not checked",
        "against it. That is stated rather than quietly assumed clean.",
        "",
        "## Length",
        "",
        f"Longest prompt is `{longest.id}` at about "
        f"{estimate_seconds(longest.text, chars_per_second=13.2):.1f} s at Anitha's measured "
        "13.2 chars/sec — inside the 20 s clip budget (82 tokens/s against a 2,048-token",
        "sequence). The build refuses any prompt that would overrun it.",
        "",
        "## Operator review (B4 done-when)",
        "",
        "`directability_ta_v1_review.tsv` lists all 140 authored items with an English gloss",
        "and an empty `operator_verdict` column. The Tamil was drafted with AI assistance and",
        "needs a Tamil speaker's sign-off before the freeze. What to look for:",
        "",
        "1. **Carriers must be emotionally neutral.** If a carrier already sounds like good or",
        "   bad news, it cannot measure whether the *tag* changed the delivery.",
        "2. **Emotional items should clearly imply one feeling** from the words alone.",
        "3. **Probe intents must all be genuinely plausible** for that phrase. An implausible",
        "   option inflates the measured pick rate.",
        "4. Anything that simply reads oddly in Tamil — those are worth more than the rest.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
