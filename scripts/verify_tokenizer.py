r"""Check the live Indic-Speak tokenizer against the ids this project compiled against.

    py -3.11 scripts\verify_tokenizer.py

``token_contract.md`` §6 requires this before data is compiled and again before training
starts, and says mismatch is "a hard error, never a warning. A silent wrong-tokenizer
run costs a full training cycle."

Run it whenever the base model is re-pulled, before a paid gate, and on a pod before the
first gradient step. It reads ``tokenizer.json`` from the Hub -- a few MB, no weights --
and exits non-zero on any disagreement.

Result on 3 Nov 2026: 28,960 added tokens, vocab 156,960, zero mismatches.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HOME", r"E:\Geetha\.hf_home")

from geetha.text.contract import TOKENIZER_FINGERPRINT  # noqa: E402
from geetha.text.tokenizer_check import (  # noqa: E402
    TokenizerMismatch,
    assert_tokenizer,
    tokenizer_mismatches,
)

REPO = "bodhan-ai/indic-speak"


def main() -> int:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "tokenizer.json")
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    added = {entry["content"]: entry["id"] for entry in document.get("added_tokens", [])}
    vocab_size = len(document["model"]["vocab"]) + len(added)

    print(f"{REPO}")
    print(f"  added tokens : {len(added):,}")
    print(f"  vocab size   : {vocab_size:,}")
    print(f"  fingerprint on record: {TOKENIZER_FINGERPRINT}")

    problems = tokenizer_mismatches(added, vocab_size)
    try:
        assert_tokenizer(added, vocab_size)
    except TokenizerMismatch as mismatch:
        print(f"\nFAIL\n{mismatch}", file=sys.stderr)
        return 1

    print(f"\nPASS - every control, conditioning and non-verbal token sits at the id this "
          f"project recorded, the audio band is complete, and the counts match. "
          f"{len(problems)} problems.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
