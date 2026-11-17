r"""Draw a deterministic sample of real Tamil sentences and show what norm_text does.

This produces the operator-review artefact for task B2: 200 sentences taken from the
corpora we actually train and test on, each with its normalised form and a flag for
whether normalisation changed it. Hand-checking real text beats hand-checking
sentences I invented, because only real text carries the punctuation, transcript
artefacts and code-mixing that the normaliser has to survive.

Reads raw\ only. Writes to work\.

    py -3.11 scripts/make_norm_review.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pyarrow.parquet as pq

from geetha.runlog import RunLog
from geetha.text.normalise import norm_text

RAW = Path(r"E:\Geetha\raw")
OUT_DIR = Path(r"E:\Geetha\work")
SEED = 1234
PER_SOURCE = 40

# (label, parquet file, text column). One shard per source is enough for a sample
# and keeps this to seconds rather than minutes.
SOURCES = [
    ("rasa_test", RAW / "hf/rasa/Tamil/test-00000-of-00008.parquet", "text"),
    # The breath/filler marks live in unsanitized_verbatim, NOT in verbatim --
    # verbatim is already sanitised down to <unintelligible>. See B2 finding F1.
    ("indicvoices_unsanitized", RAW / "hf/indicvoices/tamil/train-00000-of-00118.parquet", "unsanitized_verbatim"),
    ("indicvoices_r_test", RAW / "hf/indicvoices_r/Tamil/test-00000-of-00001.parquet", "text"),
    ("fleurs_test", RAW / "hf/fleurs/parquet-data/ta_in/test-00000-of-00001.parquet", "raw_transcription"),
    ("spring_inx_r1_test", RAW / "hf/spring_inx_tamil_r1/data/test-00000-of-00002.parquet", "text"),
]


def read_column(path: Path, column: str) -> list[str]:
    """Read one text column, never the audio column."""
    table = pq.read_table(path, columns=[column])
    return [v for v in table.column(column).to_pylist() if v]


def main() -> int:
    log = RunLog(run_id="b2-norm-review", seed=SEED, out_dir=OUT_DIR)
    log.add_param("sentences_per_source", PER_SOURCE)
    rng = random.Random(SEED)

    rows: list[dict[str, str]] = []
    for label, path, column in SOURCES:
        if not path.is_file():
            print(f"  skip {label}: {path} is not a file")
            continue
        texts = read_column(path, column)
        log.add_input(path)
        sample = rng.sample(texts, min(PER_SOURCE, len(texts)))
        print(f"  {label}: {len(texts):,} sentences available, sampled {len(sample)}")
        for text in sample:
            normalised = norm_text(text)
            rows.append(
                {
                    "source": label,
                    "original": text,
                    "normalised": normalised,
                    "changed": "yes" if normalised != text else "no",
                    "operator_verdict": "",
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "norm_review_200.tsv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "original", "normalised", "changed", "operator_verdict"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    log.add_output(out)
    log.add_param("sentences_written", len(rows))
    log.finish(status="ok")
    print(f"\nwrote {len(rows)} rows to {out}")
    print(f"changed by normalisation: {sum(r['changed'] == 'yes' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
