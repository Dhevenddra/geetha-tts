r"""The operator's monitor for a paid training run. Run it and leave it open.

    cd E:\Geetha\geetha
    .venv\Scripts\python.exe scripts\watch_run.py --run-id m2-main-1 --run-name m2-main

It prints one plain line every few minutes and ends by itself with one of two banners:

    DONE     -- training finished and the pod is gone. Report "the run is done".
    PROBLEM  -- something is wrong. Report "the run has a problem" and paste the line.

It is a reader with one exception: every pass also runs the PC watchdog
(``ops.watchdog.watch_once``), which destroys a pod that is past its deadline or not in
the burn tracker. That is the third safety layer in runbook section 9, so this window
being open is itself part of what keeps a pod from burning money unattended.

Why it reads the Hub and not the pod: everything a pod produces goes to the private
checkpoint repo, because a pod can vanish at any moment. So the truth about a run is
readable from here whether or not the pod still exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CKPT_REPO = "Dhevenddra/geetha-ckpt-private"
STATE_FILE = "trainer_state.json"
INR_PER_USD = 88.0


def stage_view(files: list[str], run_name: str, stage: str) -> tuple[str, str | None]:
    """``(status, newest_state_path)`` where status is absent / running / final."""
    prefix = f"{run_name}/stage{stage}"
    if f"{prefix}/final/{STATE_FILE}" in files:
        return "final", f"{prefix}/final/{STATE_FILE}"
    numbers = []
    for name in files:
        if name.startswith(f"{prefix}/ckpt-") and name.endswith(f"/{STATE_FILE}"):
            number = name[len(prefix) + 6:].split("/")[0]
            if number.isdigit():
                numbers.append(int(number))
    if not numbers:
        return "absent", None
    return "running", f"{prefix}/ckpt-{max(numbers):07d}/{STATE_FILE}"


def crumbs_for(files: list[str], run_id: str) -> list[str]:
    prefix = f"breadcrumbs/{run_id}/"
    return sorted(name[len(prefix):-4] for name in files
                  if name.startswith(prefix) and name.endswith(".txt"))


def verdict(*, pod_alive: bool, ever_alive: bool, crumbs: list[str],
            stage_a: str, stage_b: str) -> tuple[str, str]:
    """``(kind, message)`` with kind in running / done / problem. Pure, so it is tested."""
    failed = [c for c in crumbs if "FAILED" in c]
    bad_exit = [c for c in crumbs if "-exit-" in c and not c.endswith("-exit-0")]
    if failed:
        return "problem", f"the on-pod rehearsal failed ({failed[-1]}); the real run never started"
    if bad_exit:
        return "problem", f"a training stage ended badly ({bad_exit[-1]})"
    if stage_b == "final":
        if pod_alive:
            return "running", "training finished; the pod is saving and shutting itself down"
        return "done", "Stage A and Stage B both finished and the pod is gone"
    if not pod_alive and (ever_alive or crumbs):
        where = ("Stage B" if stage_a == "final" else "Stage A")
        return "problem", (f"the pod is gone but {where} did not finish. Nothing is billing. "
                           f"The checkpoints are safe; the run can be continued")
    if not pod_alive:
        return "running", "no pod yet (still being created, or not launched)"
    if stage_a == "final":
        return "running", "Stage B (studio voices) is training"
    if "04-starting-stage-a" in crumbs:
        return "running", "Stage A (the long one) is training"
    if crumbs:
        return "running", f"setting up ({crumbs[-1]})"
    return "running", "pod is starting"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--every", type=int, default=300, help="seconds between looks")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    from ops.runpod_client import RunPodClient
    from ops.secrets import load_secrets
    from ops.watchdog import LOG_PATH, _log, heartbeat_line, watch_once

    secrets = load_secrets()
    token = secrets.require("HF_TOKEN")
    api = HfApi(token=token)
    client = RunPodClient(secrets)
    ever_alive = False
    strikes = 0

    print(f"watching {args.run_id} (run {args.run_name}). Leave this window open. "
          f"Ctrl+C stops the watching, never the training.\n", flush=True)
    while True:
        stamp = datetime.now().strftime("%H:%M")
        try:
            now = datetime.now(timezone.utc)
            guard = watch_once(client, now=now)
            _log(heartbeat_line(now, guard), Path(LOG_PATH))
            pods = [p for p in client.live_pods() if p.name == args.run_id]
            pod_alive = bool(pods)
            ever_alive = ever_alive or pod_alive
            files = list(api.list_repo_files(CKPT_REPO, repo_type="model"))
            crumbs = crumbs_for(files, args.run_id)
            stage_a, state_a = stage_view(files, args.run_name, "A")
            stage_b, state_b = stage_view(files, args.run_name, "B")
            kind, message = verdict(pod_alive=pod_alive, ever_alive=ever_alive,
                                    crumbs=crumbs, stage_a=stage_a, stage_b=stage_b)

            detail = ""
            newest = state_b or state_a
            if newest:
                state = json.loads(Path(hf_hub_download(
                    CKPT_REPO, newest, repo_type="model", token=token)).read_text("utf-8"))
                detail = (f" | stage {state['stage']}: {state['elapsed_s'] / 3600:.1f} h trained, "
                          f"{state['tokens_seen'] / 1e6:.0f}M tokens, epoch {state['epoch'] + 1}")
            if pods:
                pod = pods[0]
                detail += f" | pod {pod.gpu_type} ${pod.cost_per_hr:.2f}/h"
            if guard.destroyed:
                kind, message = "problem", (f"the safety watchdog destroyed pod(s) "
                                            f"{', '.join(guard.destroyed)} (deadline or budget)")
            if guard.error:
                detail += f" | WATCHDOG WARNING: {guard.error}"
            strikes = 0
        except KeyboardInterrupt:
            raise
        except Exception as problem:  # noqa: BLE001 -- a network blip must not end the watch
            strikes += 1
            print(f"[{stamp}] could not look this time ({type(problem).__name__}: "
                  f"{str(problem)[:120]}); try {strikes}", flush=True)
            if strikes >= 6:
                print("\n===== PROBLEM =====\nI have not been able to look for six tries in a "
                      "row (internet, or the E: drive?). The pod has its own safety limits "
                      "and will stop itself at its deadline. Report it.\a", flush=True)
                return 2
            time.sleep(args.every)
            continue

        print(f"[{stamp}] {message}{detail}", flush=True)
        if kind == "done":
            print("\n===== DONE =====\nReport: \"the run is done\".\a", flush=True)
            return 0
        if kind == "problem":
            print(f"\n===== PROBLEM =====\n{message}\nReport: \"the run has a problem\" "
                  f"and paste the line above.\a", flush=True)
            return 2
        if args.once:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped watching. The training is not affected.")
        raise SystemExit(0)
