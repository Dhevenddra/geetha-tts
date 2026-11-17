r"""ETL step 7: push the SNAC code shards to a private HF dataset repo.

    py -3.11 scripts\push_codes_dataset.py --stage A [--dry-run]

Runbook step 7's done-when is "a pod trains without touching E:". What travels is
**codes only** (D-037): a few hundred MB of token ids against 22-88 GB of audio, and a
pod that receives codes never has to decode or resample anything, so paid GPU time is
spent on gradient steps.

What this refuses to do
-----------------------
* **Common Voice never leaves this machine** (the project rules, runbook §4). Nothing here
  reads it -- Stage A and Stage B are IV-R and Rasa only -- but the source set is
  asserted anyway, because the cost of being wrong is a licence breach that cannot be
  undone by deleting the repo.
* It will not push without the manifest's checksums present, because a code shard whose
  bytes are not recorded cannot later be tied to the numbers measured from it.
* The repo is created **private**. There is no flag here to make it public: a v1 release
  decision about publishing training data is not something a push script should be able
  to make by accident.

Licence: both sources are **CC-BY-4.0** (``manifests\licence_ledger.csv``), so
derivatives are permitted with attribution. The dataset card records that attribution,
and the model card owes AI4Bharat the same credit at release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.secrets import load_secrets  # noqa: E402

ROOT = Path(r"E:\Geetha")
CODES = ROOT / "processed" / "tts_codes"
ALLOWED_SOURCES = {"indicvoices_r", "rasa"}
FORBIDDEN = {"common_voice", "cv", "common_voice_27_ta"}

CARD = """---
license: cc-by-4.0
language: ta
tags: [tamil, tts, snac, geetha]
---

# Geetha {stage} — SNAC code shards (private)

Audio tokens for training Geetha, a Tamil TTS model. **Codes only, no audio.**

- clips: {clips:,}
- tokens: {tokens:,}
- hours of source audio represented: {hours:.1f}
- sources: IndicVoices-R and Rasa (AI4Bharat), both **CC-BY-4.0**
- codec: `hubertsiuzdak/snac_24khz`, 24 kHz mono
- layout: 7 tokens per frame, offset by frame position, per `bodhan-ai/indic-speak`
  `token_contract.md` §3. Consecutive frames sharing a `c0` are dropped.
- loudness: every clip normalised toward -23 dBFS RMS, attenuation only; the applied
  `gain_db` is a column, so any clip can be reproduced from its source.

Derived from works by **AI4Bharat** ([IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r),
[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa)), used under CC-BY-4.0.

IndicVoices-R speaker identities appear only as salted pseudonyms; the mapping is not
published. Held-out speakers and every sentence belonging to a held-out split have been
removed (leak table = 0).
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B"], default="A")
    parser.add_argument("--repo", help="override the default repo id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage_dir = CODES / f"stage{args.stage}"
    shards = sorted(stage_dir.glob("*.parquet"))
    if not shards:
        print(f"!! no shards in {stage_dir}", file=sys.stderr)
        return 1
    if not (stage_dir / "checksums.csv").exists():
        print(f"!! {stage_dir / 'checksums.csv'} is missing; a shard whose bytes are not "
              "recorded cannot be tied to a published number", file=sys.stderr)
        return 1

    clips = tokens = 0
    seconds = 0.0
    sources: set[str] = set()
    for path in shards:
        table = pq.read_table(path, columns=["source", "n_tokens", "duration_s"])
        sources.update(table["source"].to_pylist())
        clips += table.num_rows
        tokens += sum(table["n_tokens"].to_pylist())
        seconds += sum(value or 0.0 for value in table["duration_s"].to_pylist())

    stray = sources - ALLOWED_SOURCES
    if stray or sources & FORBIDDEN:
        print(f"!! REFUSING to push: unexpected source(s) {sorted(stray or sources & FORBIDDEN)}. "
              "Common Voice must never reach Hugging Face, and only IV-R and Rasa are "
              "cleared for this repo.", file=sys.stderr)
        return 1

    size_mb = sum(path.stat().st_size for path in shards) / 1e6
    repo_id = args.repo or f"Dhevenddra/geetha-tts-codes-stage{args.stage.lower()}-private"
    print(f"stage {args.stage}: {len(shards)} shards, {clips:,} clips, {tokens:,} tokens, "
          f"{seconds / 3600:.1f} h, {size_mb:.0f} MB")
    print(f"sources: {sorted(sources)}")
    print(f"target : {repo_id} (private)")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    token = load_secrets().require("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    card = stage_dir / "README.md"
    card.write_text(
        CARD.format(stage=f"Stage {args.stage}", clips=clips, tokens=tokens,
                    hours=seconds / 3600),
        encoding="utf-8")

    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["*.parquet", "checksums.csv", "README.md"],
        commit_message=f"Geetha Stage {args.stage} SNAC codes: {clips:,} clips, {tokens:,} tokens",
    )
    print(f"\npushed {len(shards)} shards to {repo_id}")
    print("the repo is PRIVATE; publishing training data is a release decision, not a push flag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
