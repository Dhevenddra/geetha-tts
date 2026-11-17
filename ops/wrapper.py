r"""The script a paid pod runs, and the two safety layers that live on the pod itself.

Runbook section 9 requires three layers on every paid pod. Two of them are here:

1. **a deadline** — the training command runs under ``timeout``, and the pod deletes
   itself when the wrapper exits for *any* reason, including a crash, an OOM, or the
   command never starting at all;
2. **an idle watchdog** — a background loop that deletes the pod if the GPU sits
   unused, which is what a silently-dead training process looks like from outside.

The third layer is the PC-side watchdog, which is off-pod on purpose and lives
elsewhere: a pod that has wedged badly enough to stop running its own wrapper cannot be
trusted to kill itself.

Why the wrapper is generated rather than written by hand
--------------------------------------------------------
Because the dangerous failure is not a pod that dies early. It is a pod that lives
forever. Every exit path has to end in a delete, and the only way to be sure of that is
to generate the script from one template and assert the property in a test, rather than
trust that each hand-edited job spec remembered the trap.

Delete, never stop
------------------
``runpodctl stop pod`` leaves the pod's disk allocated and **still billing**. Every
teardown path here is a delete. :func:`build_wrapper` is tested for the absence of any
stop call, because this is precisely the kind of mistake that looks harmless in review
and shows up as a drained balance a week later.

The undocumented bit
--------------------
RunPod injects ``RUNPOD_POD_ID`` and a pod-scoped ``RUNPOD_API_KEY`` into every pod.
Whether that scoped key is allowed to delete *its own pod* is not documented anywhere,
and the entire safety design rests on it. So the wrapper tries ``runpodctl`` first and
falls back to a REST ``DELETE``, logging both exit codes — and ``m0-selfdelete-probe``
exists to answer the question for a few rupees before any real money rides on it.
"""

from __future__ import annotations

import shlex
from textwrap import dedent

__all__ = [
    "DEFAULT_IDLE_MINUTES",
    "LOG_PUSH_SNIPPET",
    "SELF_DESTROY_SNIPPET",
    "build_probe_command",
    "build_wrapper",
]

# A training step on the smallest planned job is minutes, not tens of minutes. Twenty
# minutes of an idle GPU is therefore not a slow epoch, it is a dead process -- but it
# is long enough not to fire during a long checkpoint upload.
DEFAULT_IDLE_MINUTES = 20

# Kept as its own constant so the probe and the real wrapper test the *same* teardown
# path. A probe that proved a different code path would prove nothing.
# Ship the pod's own log to the Hub before anything deletes the pod.
#
# This is the biggest operational gap in the project. Between 3 and 5 Nov seventeen pods
# died and every one took its log with it, so each post-mortem was an inference from a
# create timestamp and a delete timestamp. That is how a bash syntax error got recorded
# as an image-pull timeout, and how a breadcrumb function that had never worked got read
# as "our code never ran". The pod knows exactly what happened; it had no way to say so.
#
# Deliberately NO heredoc and no inline python. These snippets are interpolated into an
# f-string inside dedent(), which rewrites leading whitespace -- a quoted heredoc
# terminator must sit at column 0, and an indented python block raises IndentationError.
# Single-line commands are the only shape that survives that transformation intact.
#
# Every line is best-effort: a pod that cannot upload its log must still delete itself.
LOG_PUSH_SNIPPET = dedent(
    """
    geetha_push_log() {
      echo "[geetha] pushing pod log to ${GEETHA_CKPT_REPO:-<unset>}"
      python -m pip install -q -U huggingface_hub >/dev/null 2>&1 || true
      hf upload "$GEETHA_CKPT_REPO" /workspace/pod.log "logs/${GEETHA_RUN_ID}.log" --repo-type model ||         echo "[geetha] log push failed (non-fatal)"
    }
    """
).strip()

SELF_DESTROY_SNIPPET = dedent(
    """
    geetha_self_destroy() {
      echo "[geetha] self-destroy: pod ${RUNPOD_POD_ID:-<unset>}"
      geetha_push_log || true
      runpodctl remove pod "$RUNPOD_POD_ID"
      local rc_cli=$?
      echo "[geetha] runpodctl remove exit=$rc_cli"
      if [ $rc_cli -ne 0 ]; then
        curl -sS -X DELETE \\
          -H "Authorization: Bearer $RUNPOD_API_KEY" \\
          "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID"
        echo "[geetha] REST delete exit=$?"
      fi
    }
    """
).strip()


def _idle_watchdog(idle_minutes: int) -> str:
    """Delete the pod after ``idle_minutes`` of an unused GPU.

    Utilisation is sampled rather than averaged: a process that has died leaves the GPU
    at 0%, and anything above the floor resets the counter. The floor is 5% rather than
    0% because an idle CUDA context is not always exactly zero.
    """
    checks = max(1, idle_minutes)
    return dedent(
        f"""
        geetha_idle_watchdog() {{
          local idle=0
          while true; do
            sleep 60
            local util
            util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
            util=${{util:-0}}
            if [ "$util" -lt 5 ]; then
              idle=$((idle + 1))
              echo "[geetha] GPU idle ${{idle}}/{checks} min (util ${{util}}%)"
            else
              idle=0
            fi
            if [ "$idle" -ge {checks} ]; then
              echo "[geetha] idle watchdog fired after {checks} min — destroying pod"
              geetha_self_destroy
              exit 99
            fi
          done
        }}
        """
    ).strip()


def build_wrapper(
    *,
    run_id: str,
    command: str,
    max_hours: float,
    checkpoint_repo: str,
    idle_minutes: int = DEFAULT_IDLE_MINUTES,
) -> str:
    """The full on-pod script: run the job under a deadline, then always delete the pod.

    ``set -u`` and ``set -o pipefail`` are on; ``set -e`` deliberately is **not**. An
    ``-e`` abort partway through teardown would skip the push or the delete, which is
    the one outcome worse than a failed run. Failures are checked where they matter and
    the exit code is carried to the end by hand.
    """
    if max_hours <= 0:
        raise ValueError("max_hours must be positive — a pod with no deadline is the failure mode")
    if idle_minutes <= 0:
        raise ValueError("idle_minutes must be positive")

    timeout_seconds = int(max_hours * 3600)
    LOG_PUSH = LOG_PUSH_SNIPPET
    # POSIX quoting, NOT Python's repr. `{command!r}` renders an embedded single quote
    # as ' , which is valid Python and invalid shell: inside a single-quoted string a
    # backslash is literal, so bash ends the string early. Any command containing a
    # single quote therefore became a syntax error on the pod.
    #
    # This was live from 4 Nov. The m1-pilot command quotes its repo names
    # (snapshot_download('Dhevenddra/...')), so m1-pilot-5 and m1-pilot-7 were launched
    # with a command bash could not parse -- "syntax error near unexpected token )".
    # m1-pilot-5's death was recorded as an image-pull timeout and became the premise of
    # a whole day's plan. It was this.
    #
    # shlex.quote renders the same string as '...'"'"'...', which every POSIX shell
    # parses correctly. tests/test_wrapper_quoting.py runs `bash -n` over the generated
    # script so this cannot regress by inspection alone.
    quoted_command = shlex.quote(command)
    return dedent(
        f"""
        #!/usr/bin/env bash
        # geetha wrapper for run_id={run_id} — generated, do not hand-edit on the pod.
        set -u
        set -o pipefail

        # Everything from here is teed to a file so the pod can hand us its own log on
        # the way out. Without this a dead pod is a timestamp and a guess.
        mkdir -p /workspace
        exec > >(tee -a /workspace/pod.log) 2>&1

        export GEETHA_CKPT_REPO={checkpoint_repo}
        export GEETHA_RUN_ID={run_id}

        {LOG_PUSH}

        {SELF_DESTROY_SNIPPET}

        {_idle_watchdog(idle_minutes)}

        # Whatever happens below — success, crash, OOM, the command never starting —
        # the pod is deleted on the way out.
        trap geetha_self_destroy EXIT

        geetha_idle_watchdog &
        GEETHA_WATCHDOG_PID=$!

        echo "[geetha] run_id={run_id} deadline {max_hours:g}h ({timeout_seconds}s)"
        timeout --signal=TERM --kill-after=120 {timeout_seconds} bash -lc {quoted_command}
        GEETHA_RC=$?
        echo "[geetha] command exit=$GEETHA_RC"

        kill "$GEETHA_WATCHDOG_PID" 2>/dev/null || true

        # Push whatever exists before teardown. A failed run whose checkpoint survived
        # is a bad afternoon; a failed run whose checkpoint did not is a lost gate.
        # The job fetches our package to /workspace/code, so the push needs it on the
        # path. Without this the safety net silently fails with ModuleNotFoundError --
        # seen on m1-pilot-9 -- and a run that trained for an hour would lose its
        # checkpoint at the last step, which is the one outcome worse than not starting.
        export PYTHONPATH=/workspace/code${{PYTHONPATH:+:$PYTHONPATH}}
        # `python -m geetha.ops.push_checkpoints` used to stand here. That module has
        # never existed -- not under geetha/ops/, not anywhere -- so the last safety net
        # before teardown failed on every run the project ever launched, with
        # ModuleNotFoundError. Nobody saw it because nobody could read a pod log until
        # 5 Nov. A net only exercised once something has already gone wrong has to be
        # exercised some other way, or it is decoration.
        #
        # `hf upload` is the same mechanism the log push uses, now observed working, and
        # it needs no module of ours. The training loop already uploads each checkpoint
        # as it goes; this sweeps up whatever is still on disk when the run ends.
        echo "[geetha] pushing leftover checkpoints to {checkpoint_repo}"
        if [ -d /workspace/out ] && [ -n "$(ls -A /workspace/out 2>/dev/null)" ]; then
          hf upload "{checkpoint_repo}" /workspace/out "final/{run_id}" --repo-type model || \
            echo "[geetha] WARNING: checkpoint push failed"
        else
          echo "[geetha] nothing left in /workspace/out (uploaded during training)"
        fi

        exit $GEETHA_RC
        """
    ).strip() + "\n"


def build_probe_command(sleep_seconds: int = 120) -> str:
    """``m0-selfdelete-probe`` — can a pod-scoped key delete its own pod?

    Deliberately minimal and deliberately *not* trapped: it calls the teardown path
    directly so that the thing being measured is the delete itself, not the wrapper
    around it. It logs the runpodctl version and both exit codes, and the answer is
    confirmed from the PC — the pod must end up TERMINATED, not EXITED, because a
    stopped pod still bills for its disk.
    """
    return dedent(
        f"""
        #!/usr/bin/env bash
        set -u
        echo "[probe] pod=${{RUNPOD_POD_ID:-<unset>}} key_present=$([ -n "${{RUNPOD_API_KEY:-}}" ] && echo yes || echo no)"
        runpodctl version || echo "[probe] runpodctl not installed"
        echo "[probe] sleeping {sleep_seconds}s so the pod is observable from the PC"
        sleep {sleep_seconds}

        {SELF_DESTROY_SNIPPET}

        geetha_self_destroy
        echo "[probe] if this line appears, the pod outlived its own delete call"
        sleep 600
        """
    ).strip() + "\n"
