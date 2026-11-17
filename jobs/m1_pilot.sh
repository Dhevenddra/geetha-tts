#!/usr/bin/env bash
# M1 pilot: 2 h Stage A then 2 h Stage B, on one pod.
#
# This script is the `command` that ops/wrapper.py wraps. The wrapper owns the three
# safety layers (deadline, self-destroy on any exit, on-pod idle watchdog), so nothing
# here needs to think about teardown -- and nothing here may create a background process
# that outlives it.
#
# No `set -e`. A failure partway through must still reach the result-collection step at
# the bottom, because a pilot that trained for 100 minutes and then crashed is worth
# pulling. Failures are checked where they matter and the exit code is carried by hand.
set -u
set -o pipefail

STAGE_A_SECONDS="${STAGE_A_SECONDS:-2300}"
STAGE_B_SECONDS="${STAGE_B_SECONDS:-2300}"
PILOT_SHARDS="${PILOT_SHARDS:-12}"
OUT="${OUT:-/workspace/out}"
CODE_REPO="${CODE_REPO:-Dhevenddra/geetha-code-private}"

say() { echo "[m1_pilot $(date -u +%H:%M:%S)] $*"; }

# Breadcrumbs. Four pods have now died between 4 s and 2m21s and taken their logs with
# them, so each stage reports to the checkpoint repo as it passes. A tiny file costs
# nothing and turns "it vanished" into "it got as far as X".
crumb() {
  # NOTE (5 Nov): this was a no-op on every pod ever launched. The body f-string was
  # broken across a literal newline -- "unterminated string literal" -- and `2>/dev/null
  # || true` swallowed the SyntaxError without a word. So "no breadcrumbs appeared" was
  # never evidence about how far a pod got, and it was briefly read as exactly that.
  # stderr is kept now; a breadcrumb that cannot be written must say so.
  python - "$1" <<'CRUMB' || echo "[crumb] FAILED for $1 (see error above)"
import os, sys, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("CKPT_REPO", "Dhevenddra/geetha-ckpt-private")
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
body = "{} at {} ".format(sys.argv[1], stamp).strip() + chr(10)
# Namespaced by run_id. Flat breadcrumbs/<name>.txt persist from the previous run,
# so "breadcrumb 04 exists" said nothing about THIS run -- a check for it returned
# true instantly for m1-pilot-11 while run 10's crumb was what it was reading.
run = os.environ.get("GEETHA_RUN_ID", "unknown")
api.upload_file(path_or_fileobj=body.encode(),
                path_in_repo="breadcrumbs/{}/{}.txt".format(run, sys.argv[1]),
                repo_id=repo, repo_type="model")
print("[crumb] wrote", sys.argv[1])
CRUMB
  say "breadcrumb: $1"
}


if [ -z "${HF_TOKEN:-}" ]; then
  say "FATAL: HF_TOKEN is not set; the pod cannot reach the base model or the codes"
  exit 78
fi

mkdir -p "$OUT" /workspace/code
export HF_HOME=/workspace/.hf
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

crumb 01-container-started
say "installing dependencies"
pip install -q -U "transformers>=4.44" "peft>=0.13" accelerate "huggingface_hub>=0.25" \
  pyarrow numpy soundfile wandb 2>&1 | tail -3

# Our package travels over Hugging Face, not GitHub: the pod already holds an HF token
# for the base model and the code shards, so this adds no second credential to a machine
# we do not own.
crumb 02-deps-installed
say "fetching the geetha package from $CODE_REPO"
python - <<'PY'
import os, pathlib
from huggingface_hub import snapshot_download
target = snapshot_download(os.environ.get("CODE_REPO", "Dhevenddra/geetha-code-private"),
                           repo_type="model", token=os.environ["HF_TOKEN"],
                           local_dir="/workspace/code")
print("code at", target)
assert pathlib.Path("/workspace/code/geetha/text/contract.py").exists(), \
    "the code repo does not contain geetha/text/contract.py"
PY
if [ $? -ne 0 ]; then
  say "FATAL: could not fetch the geetha package"
  exit 78
fi
export PYTHONPATH=/workspace/code
crumb 03-code-fetched

status=0

crumb 04-starting-stage-a
say "=== Stage A: ${STAGE_A_SECONDS}s, ${PILOT_SHARDS} shards ==="
python -m geetha.train.pilot --stage A \
  --max-seconds "$STAGE_A_SECONDS" --shards "$PILOT_SHARDS" \
  --lora-rank 64 --out "$OUT"
a_status=$?
say "Stage A exited $a_status"

# Stage B continues the same adapter. If Stage A produced nothing there is nothing to
# continue, so Stage B is skipped rather than started from the base model -- a Stage B
# run with no Stage A behind it is not the thing the gate is paying to test.
A_ADAPTER="$OUT/adapter_final"
if [ -d "$A_ADAPTER" ]; then
  say "=== Stage B: ${STAGE_B_SECONDS}s, continuing $A_ADAPTER ==="
  mv "$A_ADAPTER" "$OUT/adapter_stageA_final"
  python -m geetha.train.pilot --stage B \
    --max-seconds "$STAGE_B_SECONDS" --shards "$PILOT_SHARDS" \
    --lora-rank 64 --resume-from "$OUT/adapter_stageA_final" --out "$OUT"
  b_status=$?
  say "Stage B exited $b_status"
else
  say "!! Stage A left no adapter at $A_ADAPTER; skipping Stage B"
  b_status=78
fi

[ "$a_status" -ne 0 ] && status="$a_status"
[ "$b_status" -ne 0 ] && [ "$status" -eq 0 ] && status="$b_status"

say "collecting results"
ls -la "$OUT" 2>&1 | tail -20
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 | tail -2

say "done, exit $status"
exit "$status"
