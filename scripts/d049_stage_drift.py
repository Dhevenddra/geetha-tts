r"""Does Stage B corrupt the non-verbal embeddings Stage A learned?

Why this exists
---------------
D-007's recipe is two-stage: Stage A learns timing and non-verbal events from
IndicVoices phone audio, Stage B restores studio timbre on Anitha and Arun. The
manifests say what that means in practice, and it is sharper than the recipe reads:

    Stage A: 15,760 of 30,989 ivr_* rows carry a breath (27,048 breath tokens)
    Stage A: anitha 16,860 rows, arun 14,499 rows -- **zero** breaths, zero events
    Stage B: 16,530 rows, anitha and arun only -- **zero** breaths, zero events

So Stage B is the last training the model sees and contains no non-verbal token at
all. The question is whether what Stage A taught about `<|nv_breath|>` survives it.

The mechanism that makes this non-obvious
-----------------------------------------
For an embedding LoRA the effective update for token *i* is ``B @ A[:, i]`` (scaled
alpha/r = 1 here). The two factors behave completely differently under Stage B:

* ``lora_embedding_A`` is ``(r, vocab)``. Column *i* receives gradient only when token
  *i* appears. `<|nv_breath|>` never appears in Stage B, so its column is **frozen**.
* ``lora_embedding_B`` is ``(hidden, r)`` and is **shared by every token in the
  vocabulary**. Every Anitha and Arun token in Stage B updates it.

So the breath delta is ``B_after @ A_frozen``: a frozen column read through a rotated
basis. "Its own weights were not touched" does not mean "its embedding did not move".
That is the failure this measures, and it is invisible to the loss curve.

What is measured
----------------
Per token, between Stage A final and Stage B final:

* ``||B @ A[:, i]||`` -- the magnitude of the learned delta, each stage
* the **cosine** between the two stages' delta vectors -- the direction, which is what
  carries the meaning. A preserved magnitude with a rotated direction is still a loss.

Controls, taken from the D-039 check so the two measurements agree: `<|nv_burp|>`,
`<|nv_sneeze|>`, `<|pad|>` and an ordinary token all had an exactly-zero A column
after Stage A. They must stay exactly zero here. If a control moves, the instrument is
measuring something other than learning and no positive result can be trusted.

Free: reads two adapter files, no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# From results\d039_embedding_check.md, so this script and that one name the same rows.
TOKENS = {
    "<|nv_breath|>": 156_944,
    "<|nv_laugh|>": 156_947,
    "<|nv_burp|>": 156_959,
    "<|nv_sneeze|>": 156_955,
    "<|pad|> (control)": 128_263,
    "ordinary token 1000": 1_000,
}
CONTROLS = {"<|nv_burp|>", "<|nv_sneeze|>", "<|pad|> (control)", "ordinary token 1000"}

EMB_A = "base_model.model.model.embed_tokens.lora_embedding_A"
EMB_B = "base_model.model.model.embed_tokens.lora_embedding_B"


def _load(path: Path):
    """The two embedding LoRA factors, as float32 torch tensors.

    Read one tensor at a time: ``lora_embedding_A`` is 64 x 156,960 and B is 3,072 x 64,
    a few tens of MB, while the adapter file is 1.39 GB. This machine has ~1 GB of free
    RAM, so loading the whole file is not an option.
    """
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        a_key = next((k for k in keys if k.endswith("embed_tokens.lora_embedding_A")), None)
        b_key = next((k for k in keys if k.endswith("embed_tokens.lora_embedding_B")), None)
        if a_key is None or b_key is None:
            raise SystemExit(
                f"{path} has no embed_tokens LoRA. D-039 requires embed_tokens in every\n"
                f"target list; an adapter without it cannot learn a non-verbal token.\n"
                f"keys that look close: {[k for k in keys if 'embed' in k]}"
            )
        return handle.get_tensor(a_key).to(torch.float32), handle.get_tensor(b_key).to(torch.float32)


def deltas(a, b):
    """``B @ A[:, i]`` per measured token, plus how much A's column was touched."""
    import torch

    out = {}
    for name, token_id in TOKENS.items():
        column = a[:, token_id]
        out[name] = {
            "token_id": token_id,
            "delta": (b @ column),
            "norm": float(torch.linalg.vector_norm(b @ column)),
            "a_nonzero": int((column != 0).sum()),
            "a_rank": int(column.numel()),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", required=True, type=Path)
    parser.add_argument("--stage-b", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch

    a_a, a_b = _load(args.stage_a)
    b_a, b_b = _load(args.stage_b)

    # How much Stage B moved the shared factor. This is the quantity that can rotate a
    # frozen column's delta, so it is reported rather than assumed.
    shared_shift = float(torch.linalg.matrix_norm(b_b - a_b) / torch.linalg.matrix_norm(a_b))

    first = deltas(a_a, a_b)
    second = deltas(b_a, b_b)

    rows = []
    for name in TOKENS:
        one, two = first[name], second[name]
        if one["norm"] == 0.0 or two["norm"] == 0.0:
            cosine = None
        else:
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    one["delta"].unsqueeze(0), two["delta"].unsqueeze(0)
                ).item()
            )
        rows.append({
            "token": name,
            "token_id": one["token_id"],
            "stage_a_norm": one["norm"],
            "stage_b_norm": two["norm"],
            "cosine": cosine,
            "stage_a_a_nonzero": one["a_nonzero"],
            "stage_b_a_nonzero": two["a_nonzero"],
            "is_control": name in CONTROLS,
        })

    broken = [r for r in rows if r["is_control"] and (r["stage_a_norm"] or r["stage_b_norm"])]

    report = {
        "stage_a_adapter": str(args.stage_a),
        "stage_b_adapter": str(args.stage_b),
        "shared_B_relative_shift": shared_shift,
        "tokens": rows,
        "controls_clean": not broken,
    }

    print(f"shared lora_embedding_B moved {shared_shift:.4f} (relative Frobenius) in Stage B")
    print()
    print(f"{'token':24s} {'A norm':>10s} {'B norm':>10s} {'cosine':>9s}  A-col nonzero")
    for r in rows:
        cosine = "n/a" if r["cosine"] is None else f"{r['cosine']:.4f}"
        print(f"{r['token']:24s} {r['stage_a_norm']:10.6f} {r['stage_b_norm']:10.6f} "
              f"{cosine:>9s}  {r['stage_a_a_nonzero']}/{r['a_rank'] if 'a_rank' in r else 64}"
              f" -> {r['stage_b_a_nonzero']}")
    print()
    if broken:
        print("CONTROL FAILED: a token with no Stage A signal has a non-zero delta.")
        print("Nothing above can be trusted until that is explained.")
    else:
        print("controls clean: every unobserved token is exactly zero in both stages.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if not broken else 1


if __name__ == "__main__":
    sys.exit(main())
