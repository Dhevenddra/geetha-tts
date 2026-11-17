r"""Task B3 - build the dual-script references for the Tamil ASR benchmarks.

Writes ``E:\Geetha\eval_vault\refs_v1\``:

    diarbench_ta_refs.jsonl   one row per DiarBench segment, with recording_id so
                              results can be aggregated by recording (runbook rule 3)
    spring_r1_refs.jsonl      SPRING-INX R1 test, Latin words given Tamil alternatives
    spring_r2_refs.jsonl      SPRING-INX R2 test, likewise
    lexicon_ta_en.json        the Latin -> Tamil lexicon harvested from DiarBench
    refs_v1_report.md         counts, residuals and the known gaps, for the record
    spotcheck_50.tsv          50 sampled references for the operator to check (B3)

Reads ``raw\`` and ``eval_vault\`` only, and writes nothing outside ``refs_v1\``.

    py -3.11 scripts/make_refs_v1.py
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from geetha.runlog import RunLog
from geetha.scoring.dual_script import parse_alternations
from geetha.text.annotated import harvest_lexicon, parse_annotated
from geetha.text.lexicon import apply_lexicon, filter_lexicon

EVAL_VAULT = Path(r"E:\Geetha\eval_vault")
RAW = Path(r"E:\Geetha\raw")
OUT = EVAL_VAULT / "refs_v1"
SEED = 1234
MIN_LEXICON_COUNT = 2

DIARBENCH = EVAL_VAULT / "indic_diarbench" / "Tamil" / "test-00000-of-00001.parquet"
SPRING = {
    "spring_r1": sorted((RAW / "hf/spring_inx_tamil_r1/data").glob("test-*.parquet")),
    "spring_r2": sorted((RAW / "hf/spring_inx_tamil_r2/data").glob("test-*.parquet")),
}

_LEFTOVER_BRACKET = re.compile(r"[()\[\]]")
_BARE_LATIN = re.compile(r"[A-Za-z][A-Za-z']*")


def strip_alternations(text: str) -> str:
    return re.sub(r"\{[^{}]*\}", " ", text)


def build_diarbench(log: RunLog) -> tuple[list[dict], dict[str, list[str]], dict]:
    log.add_input(DIARBENCH)
    table = pq.read_table(
        DIARBENCH, columns=["recording_id", "sample_id", "annotated_transcript"]
    )

    rows: list[dict] = []
    segments = []
    pair_counts: collections.Counter[tuple[str, str]] = collections.Counter()

    for recording_id, sample_id, transcript in zip(
        table.column("recording_id").to_pylist(),
        table.column("sample_id").to_pylist(),
        table.column("annotated_transcript").to_pylist(),
    ):
        for index, seg in enumerate(transcript):
            parsed = parse_annotated(seg.get("transcript") or "")
            segments.append(parsed)
            for native, source in parsed.pairs:
                pair_counts[(source, native)] += 1
            rows.append(
                {
                    "set": "diarbench_ta",
                    "recording_id": recording_id,
                    "sample_id": sample_id,
                    "segment_index": index,
                    "speaker_id": seg.get("speaker_id"),
                    "start_time": seg.get("start_time"),
                    "end_time": seg.get("end_time"),
                    "reference": parsed.reference,
                    "tags": parsed.tags,
                    "scorable": parsed.scorable,
                }
            )

    lexicon = harvest_lexicon(segments)
    filtered = filter_lexicon(lexicon, pair_counts, min_count=MIN_LEXICON_COUNT)

    scorable = [r for r in rows if r["scorable"]]
    residual_bracket = sum(1 for r in scorable if _LEFTOVER_BRACKET.search(r["reference"]))
    residual_latin = sum(
        len(_BARE_LATIN.findall(strip_alternations(r["reference"]))) for r in scorable
    )
    stats = {
        "segments": len(rows),
        "scorable": len(scorable),
        "not_scorable": len(rows) - len(scorable),
        "recordings": len({r["recording_id"] for r in rows}),
        "tags": dict(collections.Counter(t for r in rows for t in r["tags"]).most_common()),
        "pairs": sum(pair_counts.values()),
        "lexicon_raw": len(lexicon),
        "lexicon_filtered": len(filtered),
        "residual_segments_with_bracket": residual_bracket,
        "residual_bare_latin_tokens": residual_latin,
    }
    return rows, filtered, stats


def build_spring(log: RunLog, name: str, paths: list[Path], lexicon: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    enriched = 0
    unexpandable = 0
    for path in paths:
        log.add_input(path)
        table = pq.read_table(path, columns=["utterance_id", "text"])
        for utterance_id, text in zip(
            table.column("utterance_id").to_pylist(), table.column("text").to_pylist()
        ):
            original = text or ""
            try:
                reference = apply_lexicon(original, lexicon)
            except ValueError:
                # Source text already contains { | } -- keep it verbatim rather than
                # guess, and count it.
                reference, unexpandable = original, unexpandable + 1
            if reference != original:
                enriched += 1
            rows.append(
                {
                    "set": name,
                    "utterance_id": utterance_id,
                    "reference": reference,
                    "sites": sum(1 for p in parse_alternations(reference) if isinstance(p, list)),
                    "scorable": bool(reference.strip()),
                }
            )

    latin_left = sum(
        len(_BARE_LATIN.findall(strip_alternations(r["reference"]))) for r in rows
    )
    stats = {
        "utterances": len(rows),
        "references_given_tamil_alternatives": enriched,
        "max_alternation_sites": max((r["sites"] for r in rows), default=0),
        "refused": unexpandable,
        "bare_latin_tokens_remaining": latin_left,
    }
    return rows, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_id="b3-refs-v1", seed=SEED, out_dir=OUT)
    log.add_param("min_lexicon_count", MIN_LEXICON_COUNT)

    print("DiarBench Tamil...")
    diar_rows, lexicon, diar_stats = build_diarbench(log)
    write_jsonl(OUT / "diarbench_ta_refs.jsonl", diar_rows)
    (OUT / "lexicon_ta_en.json").write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for key, value in diar_stats.items():
        print(f"   {key}: {value}")

    spring_stats = {}
    all_spring: list[dict] = []
    for name, paths in SPRING.items():
        if not paths:
            print(f"{name}: no test parquet found, skipped")
            continue
        print(f"{name}...")
        rows, stats = build_spring(log, name, paths, lexicon)
        write_jsonl(OUT / f"{name}_refs.jsonl", rows)
        all_spring.extend(rows)
        spring_stats[name] = stats
        for key, value in stats.items():
            print(f"   {key}: {value}")

    # Operator spot-check: 50 references across all three sets (B3 done-when).
    rng = random.Random(SEED)
    pool = [r for r in diar_rows if r["scorable"]] + [r for r in all_spring if r["scorable"]]
    sample = rng.sample(pool, min(50, len(pool)))
    with (OUT / "spotcheck_50.tsv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("set\tid\treference\talternation_sites\toperator_verdict\n")
        for row in sample:
            ident = row.get("utterance_id") or f"{row['recording_id']}#{row['segment_index']}"
            sites = row.get("sites")
            if sites is None:
                sites = sum(
                    1 for part in parse_alternations(row["reference"]) if isinstance(part, list)
                )
            handle.write(f"{row['set']}\t{ident}\t{row['reference']}\t{sites}\t\n")

    report = OUT / "refs_v1_report.md"
    report.write_text(_report(diar_stats, spring_stats), encoding="utf-8")

    for name in (
        "diarbench_ta_refs.jsonl",
        "lexicon_ta_en.json",
        "spotcheck_50.tsv",
        "refs_v1_report.md",
    ):
        log.add_output(OUT / name)
    for name in spring_stats:
        log.add_output(OUT / f"{name}_refs.jsonl")
    log.finish(status="ok")
    print(f"\nwrote {OUT}")
    return 0


def _report(diar: dict, spring: dict) -> str:
    lines = [
        "# refs_v1 - dual-script references for the Tamil ASR benchmarks",
        "",
        "Built by `scripts/make_refs_v1.py` (task B3), seed 1234. Provenance and input",
        "hashes are in `b3-refs-v1.runlog.json`.",
        "",
        "## What a reference looks like",
        "",
        "Code-mixed spans carry every rendering a Tamil transcriber would legitimately",
        "write, as `{reading|reading}` alternations. The scorer takes whichever reading",
        "the hypothesis matches and records which one it used, so a correct",
        "transcription is never penalised for a script choice. Example:",
        "",
        "    {ஓகே|okay} {மேம்|ma'am} {டூ|two} நிமிஷம்",
        "",
        "## DiarBench Tamil",
        "",
        "Its transcripts already annotate every loanword inline (`நேச்சுரல்(Natural)`),",
        "so the references are harvested from the corpus, not transliterated.",
        "",
    ]
    for key, value in diar.items():
        if key == "tags":
            continue
        lines.append(f"- **{key}**: {value}")
    lines += ["", "Nonverbal tags found (removed from references, kept as labels):", ""]
    lines += [f"- `{tag}`: {count}" for tag, count in diar["tags"].items()]
    lines += [
        "",
        "## SPRING-INX tests",
        "",
        "These write English in Latin with no Tamil form, so the DiarBench lexicon is",
        f"applied to them. Spellings seen fewer than {MIN_LEXICON_COUNT} times are dropped:",
        "the corpus contains genuine annotation errors, and importing those would hand",
        "us free credit on a published benchmark.",
        "",
    ]
    for name, stats in spring.items():
        lines.append(f"### {name}")
        lines += [f"- **{k}**: {v}" for k, v in stats.items()]
        lines.append("")
    lines += [
        "## Known gaps, stated rather than hidden",
        "",
        "1. **A residual of source irregularities.** About 0.3% of DiarBench segments",
        "   keep a stray bracket from malformed annotation (mismatched brackets, a",
        "   bracket with no Tamil word before it). Counted above, not silently dropped.",
        "2. **The lexicon is one corpus's spelling habits.** It comes from DiarBench",
        "   conversational Tamil and may not cover SPRING-INX's domain evenly; the",
        "   per-set coverage counts above show how far it reached.",
        "",
        "## Operator check (B3 done-when)",
        "",
        "`spotcheck_50.tsv` holds 50 sampled references with an empty `operator_verdict`",
        "column. The question for each: *would a Tamil transcriber accept every reading",
        "offered here as a correct transcription of that utterance?* A reading that is",
        "wrong makes the benchmark too lenient, which is the failure that matters.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
