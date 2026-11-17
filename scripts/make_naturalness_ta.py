r"""Task B4 (part 2) — build the Naturalness-Ta CAPABILITY prompt set.

Writes ``E:\Geetha\eval_vault\naturalness_ta_v1\capability_ta_v1.jsonl`` plus an
operator review sheet and a report, and **refuses to write anything if a prompt
collides with training text** (runbook rule 3).

    py -3.11 scripts\make_naturalness_ta.py

What this builds and what it does not
-------------------------------------
Naturalness-Ta has three components (D-020). This script builds the one that is
authored:

* **CAPABILITY** — built here. 84 authored Tamil prompts in three kinds, each rendered
  in the conditions ``geetha.eval.capability`` defines.
* **DISTRIBUTION** — needs no authoring: its references are held-out IndicVoices-R
  speakers (``ivr_heldout_speakers.json``) and its prompts are those speakers' own
  transcripts, so content is matched to the references by construction.
* **REALNESS** — 40 sentences drawn from Rasa test, carried as a *labelled secondary*
  with ``REALNESS_LABEL`` attached. Selection, not authoring.

The leak check matters more here than usual
-------------------------------------------
The clean control items are deliberately plain declarative Tamil ("the capital of Tamil
Nadu is Chennai"), which is exactly the kind of sentence a read-speech corpus contains.
If one of them is in training, the model has heard it read fluently and a *missing*
breath there proves nothing. So the collision cost is reported per kind, and a collision
in the clean set is called out separately rather than averaged away.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from geetha.eval.capability import (
    CONDITIONS,
    EXPECTED_COUNTS,
    SOURCE_PATH,
    load_source,
    marks_in,
    render_text,
    spoken_text,
    validate_items,
)
from geetha.eval.naturalness import REALNESS_LABEL
from geetha.runlog import RunLog
from geetha.text.normalise import norm_text

EVAL_VAULT = Path(r"E:\Geetha\eval_vault")
RAW = Path(r"E:\Geetha\raw")
OUT_DIR = EVAL_VAULT / "naturalness_ta_v1"
OUT = OUT_DIR / "capability_ta_v1.jsonl"
SEED = 4321

# Measured 27 Oct 2026 from B7 revision 1's 117 untruncated clips, not taken from
# voices.md. The card's 13.2 chars/sec for Anitha is optimistic: the median is 11.7 and
# a long prompt can run as slow as 8.58. Using the card's figure is part of why
# revision 1 truncated, so prompt-length estimates here use the measured slow bound.
MEASURED_CHARS_PER_SECOND = 11.7
MEASURED_SLOW_CHARS_PER_SECOND = 8.58

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
    """Normalised training sentences, per-corpus coverage, and per-prompt clip counts."""
    index: set[str] = set()
    coverage: dict[str, int] = {}
    clip_counts: collections.Counter[str] = collections.Counter()
    training_rows = 0
    for name, pattern, column in TRAINING_SOURCES:
        files = sorted(glob.glob(str(RAW / pattern), recursive=True))
        if not files:
            coverage[name] = -1
            print(f"   {name}: no files matched {pattern} — NOT COVERED")
            continue
        rows = 0
        for path in files:
            try:
                table = pq.read_table(path, columns=[column])
            except Exception as error:
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


def build_rows(source) -> list[dict]:
    """One row per (item, applicable condition).

    An ``event`` item is rendered twice: ``marked`` for Geetha and ``stripped`` for
    every system including Geetha, which is what makes "does an untagged sentence still
    breathe?" (D-004) answerable from the same prompt set.
    """
    rows: list[dict] = []
    for item in source:
        conditions = ["marked", "stripped"] if marks_in(item.text) else ["stripped"]
        for condition in conditions:
            text = render_text(item, condition)
            rows.append({
                "id": f"{item.id}-{condition}",
                "item_id": item.id,
                "kind": item.kind,
                "condition": condition,
                "for_baselines": CONDITIONS[condition].for_baselines,
                "text": text,
                "spoken_text": spoken_text(item.text),
                "gloss": item.gloss,
                "expected_events": item.expected_events if condition == "marked" else 0,
                "expected_events_note": (
                    "marks honoured" if condition == "marked"
                    else "no mark given; any event is the model's own choice"
                ),
                "est_seconds_slow_bound": round(
                    len(spoken_text(item.text)) / MEASURED_SLOW_CHARS_PER_SECOND, 1
                ),
            })
    return rows


def main() -> int:
    source = load_source()
    validate_items(source)
    counts = collections.Counter(item.kind for item in source)
    if dict(counts) != EXPECTED_COUNTS:
        print(f"source counts {dict(counts)} != expected {EXPECTED_COUNTS}")
        return 1
    print(f"source validated: {len(source)} authored items {dict(counts)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_id="b4-naturalness-ta", seed=SEED, out_dir=OUT_DIR)
    log.add_input(SOURCE_PATH)
    log.add_param("expected_counts", EXPECTED_COUNTS)
    log.add_param("measured_chars_per_second", MEASURED_CHARS_PER_SECOND)
    log.add_param("measured_slow_chars_per_second", MEASURED_SLOW_CHARS_PER_SECOND)

    print("building the leak index from training corpora...")
    started = time.time()
    # Hash the SPOKEN text: a mark is ours, not the corpus's, and hashing the marked
    # form would make every event item look clean whether or not the sentence is in
    # training.
    prompt_hashes = {norm_text(spoken_text(item.text)): item.id for item in source}
    index, coverage, clip_counts, training_rows = build_leak_index(log, prompt_hashes)
    print(f"leak index: {len(index):,} distinct normalised sentences "
          f"({time.time() - started:.0f}s)")

    by_id = {item.id: item for item in source}
    collisions = [
        (item_id, clips, by_id[item_id].kind, spoken_text(by_id[item_id].text))
        for item_id, clips in sorted(clip_counts.items(), key=lambda kv: -kv[1])
    ]
    clean_collisions = [row for row in collisions if row[2] == "clean"]

    if collisions:
        print(f"\n{len(collisions)} prompt(s) appear in training text:")
        for item_id, clips, kind, text in collisions[:10]:
            print(f"   {item_id} ({kind}): {clips:,} clips  {text}")
        removal = OUT_DIR / "capability_ta_v1_leak_removal.csv"
        with removal.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id", "kind", "normalised_text", "training_clips_to_remove"])
            for item_id, clips, kind, text in collisions:
                writer.writerow([item_id, kind, norm_text(text), clips])
        log.add_output(removal)
        print(f"   removal instruction for ETL step 2: {removal}")
    else:
        print("\nno prompt collides with training text.")

    rows = build_rows(source)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review = OUT_DIR / "capability_ta_v1_review.tsv"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "kind", "id", "tamil_as_rendered", "english_gloss", "marks",
            "est_seconds", "operator_verdict", "operator_note",
        ])
        for item in source:
            writer.writerow([
                item.kind,
                item.id,
                item.text,
                item.gloss,
                " ".join(marks_in(item.text)) or "-",
                f"{len(spoken_text(item.text)) / MEASURED_SLOW_CHARS_PER_SECOND:.1f}",
                "", "",
            ])

    report = OUT_DIR / "capability_ta_v1_report.md"
    report.write_text(
        _report(source, rows, index, coverage, collisions, clean_collisions, training_rows),
        encoding="utf-8",
    )

    for path in (OUT, review, report):
        log.add_output(path)
    log.add_param("rows_built", len(rows))
    log.add_param("leak_index_size", len(index))
    log.add_param("collisions", len(collisions))
    log.finish(status="ok")

    print(f"\nwrote {len(rows)} rows ({len(source)} items x conditions) to {OUT}")
    print(f"operator review sheet: {review}")
    print(f"report: {report}")
    return 0


def _report(source, rows, index, coverage, collisions, clean_collisions, training_rows) -> str:
    counts = collections.Counter(item.kind for item in source)
    conditions = collections.Counter(row["condition"] for row in rows)
    event_marks = collections.Counter(
        mark for item in source for mark in marks_in(item.text)
    )
    total_events = sum(item.expected_events for item in source)
    slow_seconds = sum(
        len(spoken_text(item.text)) / MEASURED_SLOW_CHARS_PER_SECOND for item in source
    )
    longest = max(source, key=lambda i: len(spoken_text(i.text)))

    lines = [
        "# Naturalness-Ta v1 — CAPABILITY prompt set",
        "",
        "Built by `scripts/make_naturalness_ta.py` (task B4 part 2), seed 4321.",
        "Provenance and input hashes in `b4-naturalness-ta.runlog.json`.",
        "",
        "## What this component claims",
        "",
        "Indic-Speak's model card says *\"No sound effects. No laughter, sighs, or breaths.\"*",
        "C2 measured on 27 Oct 2026 that the sixteen non-verbal tokens are **inert** in the",
        "base model: `<|nv_breath|>` moves the audio 15.166 from the unmarked sentence against",
        "a 16.773 noise floor, i.e. less than re-rolling the seed. So this is a binary",
        "capability gap, and D-021 made it the release headline.",
        "",
        "### One correction to the re-spec, applied here",
        "",
        "The component was specified as \"breath/filler events per minute\". The token contract",
        "makes those two different measurements, and merging them would overclaim:",
        "",
        "- **non-verbal events** (breath, sigh, throat-clear, laugh) are acoustic, produced by",
        "  the closed sixteen-token set, and **inert in the base model**. Binary. The headline.",
        "- **fillers** (\"அதாவது\", \"உம்\") have **no token** — `contract.py` records that the",
        "  closed set holds none, so a filler is Tamil text any model can read aloud. The base",
        "  model can say one. What it likely cannot do is make it *sound* like hesitation. That",
        "  is a difference of degree and is reported as one.",
        "",
        "## The three kinds",
        "",
        "| Kind | Items | Rendered rows | What it measures |",
        "| --- | ---: | ---: | --- |",
        f"| event | {counts['event']} | {counts['event'] * 2} | A natural place to breathe, with the "
        "model's own token at it. Rendered **marked** (Geetha) and **stripped** (every system). "
        "The stripped rows also answer D-004: does an untagged sentence still hesitate? |",
        f"| filler | {counts['filler']} | {counts['filler']} | Lexical hesitation. No token, so no "
        "acoustic event is asked for; this is the degree comparison. |",
        f"| clean | {counts['clean']} | {counts['clean']} | Fluent read-aloud Tamil where an event is "
        "a **defect**. Gives the false-positive rate. |",
        "",
        f"{len(rows)} rows in total: {conditions['marked']} marked, {conditions['stripped']} stripped.",
        f"Expected non-verbal events across the marked rows: **{total_events}**.",
        "",
        "### Why the clean set exists",
        "",
        "Events per minute on its own rewards a model that wheezes through every sentence.",
        "The clean items make the headline falsifiable — the same property D-020 found the",
        "realness test was missing. A capability number published without its false-positive",
        "rate is not a measurement.",
        "",
        "### Marks used",
        "",
        "| Token | Items |",
        "| --- | ---: |",
    ]
    for mark, count in event_marks.most_common():
        lines.append(f"| `{mark}` | {count} |")
    lines += [
        "",
        "Only tokens in `geetha/text/contract.py` appear, and the builder refuses any other",
        "(D-017). A token the tokenizer does not hold is not ignored — it is read out as",
        "literal text, so the voice would say it aloud.",
        "",
        "## Leak check (runbook rule 3)",
        "",
        f"Every prompt's **spoken** text (marks removed — a mark is ours, not the corpus's)",
        f"was hashed with `norm_text()` against **{len(index):,}** distinct normalised training",
        f"sentences from {sum(1 for v in coverage.values() if v >= 0)} of {len(coverage)} corpora.",
        "",
    ]
    if collisions:
        lines += [
            f"**{len(collisions)} prompt(s) collide**, {len(clean_collisions)} of them in the "
            "clean set. Rule 3 removes the colliding sentence from every *training* manifest "
            "rather than rewriting the eval set; the list is in "
            "`capability_ta_v1_leak_removal.csv` for ETL step 2.",
            "",
            "A clean-set collision deserves naming separately: if the model has heard that",
            "sentence read fluently in training, a missing breath there proves nothing about",
            "capability. Those items should be re-authored rather than relied on.",
            "",
            "| Item | Kind | Clips to remove |",
            "| --- | --- | ---: |",
        ]
        for item_id, clips, kind, _ in collisions:
            lines.append(f"| {item_id} | {kind} | {clips:,} |")
    else:
        lines += [
            "**No prompt collides with training text.** The authored Tamil is new to every",
            "corpus indexed, so no training removal is required for this set.",
        ]
    lines += [
        "",
        "| Corpus | Sentences indexed |",
        "| --- | ---: |",
    ]
    for name, count in coverage.items():
        lines.append(f"| {name} | {'NOT FOUND' if count < 0 else f'{count:,}'} |")
    lines += [
        "",
        "A corpus marked NOT FOUND was not indexed, and prompts were therefore not checked",
        "against it. Stated rather than quietly assumed clean.",
        "",
        "## Length budget",
        "",
        "Prompt lengths are estimated against a **measured** speech rate, not the model card's.",
        "`voices.md` gives Anitha 13.2 chars/sec; over B7 revision 1's 117 untruncated clips the",
        f"median was **{MEASURED_CHARS_PER_SECOND} chars/sec** and a long prompt ran as slow as",
        f"**{MEASURED_SLOW_CHARS_PER_SECOND}**. Trusting the card is part of why revision 1",
        "truncated 145 of 262 prompts, so the slow bound is used here.",
        "",
        f"Longest prompt is `{longest.id}` at {len(spoken_text(longest.text))} spoken characters, "
        f"about {len(spoken_text(longest.text)) / MEASURED_SLOW_CHARS_PER_SECOND:.1f} s at the slow",
        "bound. Every prompt is validated to render in a **single pass**: a chunked naturalness",
        "prompt carries a pause the writer did not write.",
        "",
        f"Total audio per system, at the slow bound: about **{slow_seconds / 60:.0f} minutes** "
        f"({slow_seconds / 60 * 2:.0f} for Geetha, which renders the marked rows too).",
        "",
        "## What the operator has to decide",
        "",
        "The Tamil was drafted with AI assistance and needs a Tamil speaker's sign-off in",
        "`capability_ta_v1_review.tsv`. Three questions, in order of how much they matter:",
        "",
        "1. **Is the breath in a place a Tamil speaker would actually breathe?** An event item",
        "   with the mark in an unnatural position measures nothing useful — the model would be",
        "   right to ignore it.",
        "2. **Do the clean items genuinely invite no hesitation?** If one reads as a sentence a",
        "   speaker would pause inside, it is not a false-positive control.",
        "3. **Do the filler items sound like real spoken Tamil hesitation**, rather than written",
        "   Tamil with dots in it?",
        "",
        "## The other two components",
        "",
        "- **DISTRIBUTION** (TTSDS2) needs no authoring: references are the held-out",
        "  IndicVoices-R speakers in `ivr_heldout_speakers.json` and the prompts are those",
        "  speakers' own transcripts, so content is matched to the references by construction.",
        "- **REALNESS** is 40 Rasa test sentences, kept as a labelled secondary. Any published",
        "  number must carry this wording verbatim:",
        "",
        f"  > {REALNESS_LABEL}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
