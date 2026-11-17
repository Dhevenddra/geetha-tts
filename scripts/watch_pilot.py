r"""Watch a running pilot and pull its adapter the moment one lands.

    py -3.11 scripts\watch_pilot.py --run-id m1-pilot-10
    py -3.11 scripts\watch_pilot.py --run-id m1-pilot-10 --once    # just report, don't wait

Why this exists
---------------
Everything a pod produces goes to the private checkpoint repo on the Hub, not to the
pod's own disk, because a pod can vanish at any moment and on this project they
routinely have. That means the run's state is readable from anywhere, by anyone with
the token, at any time -- during the run, after the pod is destroyed, tomorrow.

So this script is deliberately a *reader*. It starts nothing, spends nothing, and
changes nothing on the Hub. Run it, interrupt it, run it again next week; it reports
the same truth each time. If a session ends mid-run -- a usage limit, a closed laptop,
a reboot -- nothing is lost and nothing needs restarting: the pod keeps training, keeps
pushing checkpoints every 250 steps, and this script will find them later.

What it does NOT protect against
--------------------------------
A pod killed abruptly (Runpod balance hitting zero, host failure) may not run the
wrapper's EXIT trap, so the *final* checkpoint push can be skipped. The periodic
step checkpoints already on the Hub survive that, so the worst case is losing up to
`--checkpoint-every` steps, not the run. "Resumable", not "lossless".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ops.secrets import load_secrets  # noqa: E402

CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
RUNS_ROOT = Path(r"E:\Geetha\runs")

# The job script's breadcrumbs, in the order they are written. Reading them tells you
# how far a run got even when the pod is long gone.
STAGES = [
    ("01-container-started", "container up"),
    ("02-deps-installed", "dependencies installed"),
    ("03-code-fetched", "geetha package fetched"),
    ("04-starting-stage-a", "Stage A training"),
    ("05-stage-a-done", "Stage A finished"),
    ("06-starting-stage-b", "Stage B training"),
    ("07-done", "run complete"),
]


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=load_secrets().require("HF_TOKEN"))


def _files(api) -> list[str]:
    try:
        return api.list_repo_files(CKPT_REPO, repo_type="model")
    except Exception as problem:  # a transient Hub error is not "the run died"
        print(f"   (could not read the repo: {type(problem).__name__}; will retry)")
        return []


def report(api, run_id: str) -> tuple[bool, list[str]]:
    """Print how far the run got. Returns (a final adapter exists, adapter dirs)."""
    files = _files(api)
    crumbs = {f.split("/")[-1].replace(".txt", "") for f in files if f.startswith("breadcrumbs/")}

    reached = [label for name, label in STAGES if name in crumbs]
    print(f"  progress: {reached[-1] if reached else 'nothing yet'}"
          f"   ({len(reached)}/{len(STAGES)} breadcrumbs)")

    # Checkpoints land under stageA/stepN and stageB/stepN -- pilot.py's save() passes
    # path_in_repo=f"stage{args.stage}/{tag}". An earlier version of this script looked
    # for an "adapter_" prefix (the pod's LOCAL directory name) and so reported "no
    # adapters" while ten of them sat on the Hub. The pod's layout is not the repo's.
    adapters = sorted({"/".join(f.split("/")[:2]) for f in files
                       if f.startswith(("stageA/", "stageB/", "final/"))})
    if adapters:
        print(f"  adapters on the Hub: {', '.join(adapters)}")
    logs = [f for f in files if f.startswith("logs/") and run_id in f]
    if logs:
        print(f"  pod log: {logs[0]}")
    return any("final" in a for a in adapters), adapters


def pull(api, run_id: str, adapters: list[str], all_checkpoints: bool = False) -> int:
    """Download every adapter to runs\\<run_id>\\. Reading only; the Hub is untouched."""
    from huggingface_hub import snapshot_download

    target = RUNS_ROOT / run_id
    target.mkdir(parents=True, exist_ok=True)
    # Only the newest checkpoint by default: they are ~2.1 GB each (embed_tokens is
    # in the LoRA targets per D-039), so pulling ten is 21 GB of near-identical weights.
    newest = max(adapters, key=lambda a: int("".join(c for c in a.split("/")[-1] if c.isdigit()) or 0))
    chosen = adapters if all_checkpoints else [newest]
    print(f"  pulling: {', '.join(chosen)}")
    patterns = [f"{a}/*" for a in chosen] + ["logs/*", "breadcrumbs/*"]
    snapshot_download(CKPT_REPO, repo_type="model", local_dir=str(target),
                      allow_patterns=patterns, token=load_secrets().require("HF_TOKEN"))
    got = [p for p in target.rglob("*") if p.is_file()]
    print(f"\npulled {len(got)} files -> {target}")
    for a in chosen:
        safet = list((target / a).glob("*.safetensors")) + list((target / a).glob("*.bin"))
        size = sum(p.stat().st_size for p in safet) / 1e6
        print(f"  {a:28s} {len(safet)} weight file(s), {size:.1f} MB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--once", action="store_true",
                        help="report and exit; do not wait for the run to finish")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="pull every step checkpoint, not just the newest (~2.1 GB each)")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    args = parser.parse_args()

    api = _api()
    print(f"watching {args.run_id} on {CKPT_REPO}")

    deadline = time.time() + args.max_minutes * 60
    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}]")
        done, adapters = report(api, args.run_id)
        if args.once:
            return pull(api, args.run_id, adapters, args.all_checkpoints) if adapters else 0
        if done:
            print("\nfinal adapter is on the Hub — pulling.")
            return pull(api, args.run_id, adapters, args.all_checkpoints)
        if time.time() >= deadline:
            print("\nstopped watching (max-minutes reached). The run is unaffected: "
                  "re-run this script any time, the Hub keeps the state.")
            return pull(api, args.run_id, adapters, args.all_checkpoints) if adapters else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
