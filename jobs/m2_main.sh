#!/usr/bin/env bash
# M2 main run: Stage A at full length, then Stage B, on one pod (D-051).
#
# This script is the `command` that ops/wrapper.py wraps. The wrapper owns the three
# safety layers (deadline, self-destroy on any exit, on-pod idle watchdog), so nothing
# here thinks about teardown, and nothing here may leave a background process behind.
#
# RELAUNCHING IS SAFE AND IS THE RECOVERY PATH. geetha.train.stage looks in the
# checkpoint repo under $RUN_NAME: a finished stage is skipped, an unfinished one
# continues from its last checkpoint at the same epoch and batch. If a pod dies, launch
# the same job again under a new run_id with the same RUN_NAME.
#
# No `set -e`: a failure partway must still reach the result listing at the bottom.
set -u
set -o pipefail

RUN_NAME="${RUN_NAME:-m2-main}"
STAGE_A_SECONDS="${STAGE_A_SECONDS:-32400}"     # 9.0 h
STAGE_B_SECONDS="${STAGE_B_SECONDS:-10800}"     # 3.0 h
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-2}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-1}"
BATCH_TOKENS="${BATCH_TOKENS:-6000}"
SHARDS="${SHARDS:-0}"
OUT="${OUT:-/workspace/out}"
CODE_REPO="${CODE_REPO:-Dhevenddra/geetha-code-private}"

say() { echo "[m2_main $(date -u +%H:%M:%S)] $*"; }

crumb() {
  python - "$1" <<'CRUMB' || echo "[crumb] FAILED for $1 (see error above)"
import os, sys, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("CKPT_REPO", "Dhevenddra/geetha-ckpt-private")
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
body = "{} at {} ".format(sys.argv[1], stamp).strip() + chr(10)
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

crumb 01-container-started
say "installing dependencies"
# peft is pinned to the version the pilot trained under and D-049 was measured under.
pip install -q -U "transformers>=5.0" "peft==0.21.2" accelerate "huggingface_hub>=0.25" \
  pyarrow numpy soundfile wandb 2>&1 | tail -3
python - <<'PY'
import torch, transformers, peft, huggingface_hub
print("versions: torch", torch.__version__, "transformers", transformers.__version__,
      "peft", peft.__version__, "hub", huggingface_hub.__version__)
print("gpu:", torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB",
      "bf16", torch.cuda.is_bf16_supported())
PY
crumb 02-deps-installed

say "fetching the geetha package from $CODE_REPO"
python - <<'PY'
import os, pathlib
from huggingface_hub import snapshot_download
target = snapshot_download(os.environ.get("CODE_REPO", "Dhevenddra/geetha-code-private"),
                           repo_type="model", token=os.environ["HF_TOKEN"],
                           local_dir="/workspace/code")
print("code at", target)
assert pathlib.Path("/workspace/code/geetha/train/stage.py").exists(), \
    "the code repo does not contain geetha/train/stage.py"
PY
if [ $? -ne 0 ]; then
  say "FATAL: could not fetch the geetha package"
  exit 78
fi
export PYTHONPATH=/workspace/code
crumb 03-code-fetched

# ---- rehearsal --------------------------------------------------------------------
# The whole pipeline in miniature, on the real model and the real GPU, before the real
# run: Stage A -> checkpoint -> final -> Stage B with replay -> final. Stage B's start is
# otherwise first exercised nine hours in, and every launch bug this project has had was
# found by launching. Five minutes here is about INR 5; finding the same bug at hour
# nine is a relaunch and a day.
if [ "${REHEARSAL:-1}" = "1" ]; then
  REH_NAME="rehearsal/${GEETHA_RUN_ID:-unknown}"
  REH_OUT=/workspace/rehearsal
  say "=== rehearsal: both stages, ~75 s each, as ${REH_NAME} ==="
  python -m geetha.train.stage --stage A --run-name "$REH_NAME" \
    --planned-seconds 75 --max-epochs 1 --warmup-steps 5 --shards 6 \
    --batch-tokens "$BATCH_TOKENS" --grad-accum 3 --checkpoint-minutes 0.5 --out "$REH_OUT" \
  && python -m geetha.train.stage --stage B --run-name "$REH_NAME" --init-from-stage A \
    --planned-seconds 75 --max-epochs 1 --warmup-steps 5 --shards 6 \
    --learning-rate 5e-5 --replay-fraction 0.2 \
    --batch-tokens "$BATCH_TOKENS" --grad-accum 3 --checkpoint-minutes 0.5 --out "$REH_OUT"
  reh_status=$?
  rm -rf "$REH_OUT"
  if [ "$reh_status" -ne 0 ]; then
    say "FATAL: the rehearsal failed (exit $reh_status); not starting the real run"
    crumb "04-rehearsal-FAILED-$reh_status"
    exit 79
  fi
  crumb 04-rehearsal-passed
fi

crumb 04-starting-stage-a
say "=== Stage A: ${STAGE_A_SECONDS}s or ${STAGE_A_EPOCHS} epochs, run ${RUN_NAME} ==="
python -m geetha.train.stage --stage A --run-name "$RUN_NAME" \
  --planned-seconds "$STAGE_A_SECONDS" --max-epochs "$STAGE_A_EPOCHS" \
  --learning-rate 1e-4 --warmup-steps 100 \
  --batch-tokens "$BATCH_TOKENS" --grad-accum 3 --shards "$SHARDS" \
  --nv-dropout 0.3 --style-dropout 0.1 --env-dropout 0.15 --out "$OUT"
a_status=$?
say "Stage A exited $a_status"
crumb "05-stage-a-exit-$a_status"

# Stage B only ever continues a FINISHED Stage A. A Stage A that diverged or ran out of
# memory exits non-zero and stores no `final`, and tuning the studio voices on top of a
# broken adapter is not what the gate is paying for.
b_status=78
if [ "$a_status" -eq 0 ]; then
  say "=== Stage B: ${STAGE_B_SECONDS}s or ${STAGE_B_EPOCHS} epochs, with replay ==="
  python -m geetha.train.stage --stage B --run-name "$RUN_NAME" --init-from-stage A \
    --planned-seconds "$STAGE_B_SECONDS" --max-epochs "$STAGE_B_EPOCHS" \
    --learning-rate 5e-5 --warmup-steps 50 --replay-fraction 0.2 \
    --batch-tokens "$BATCH_TOKENS" --grad-accum 3 --shards "$SHARDS" \
    --nv-dropout 0.3 --style-dropout 0.1 --env-dropout 0.15 --out "$OUT"
  b_status=$?
  say "Stage B exited $b_status"
  crumb "06-stage-b-exit-$b_status"
else
  say "!! Stage A did not finish cleanly; skipping Stage B"
fi

status=0
[ "$a_status" -ne 0 ] && status="$a_status"
[ "$b_status" -ne 0 ] && [ "$status" -eq 0 ] && status="$b_status"

say "collecting results"
ls -la "$OUT" 2>&1 | tail -20
rm -rf "$OUT/resume" "$OUT"/ckpt_* 2>/dev/null
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>&1 | tail -2

say "done, exit $status"
exit "$status"
