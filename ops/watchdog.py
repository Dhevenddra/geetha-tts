r"""The third safety layer: a watchdog that does not live on the pod.

    py -3.11 -m ops.watchdog            # loop, writing E:\Geetha\logs\gpu_watch.log
    py -3.11 -m ops.watchdog --once     # one pass, for a scheduled task

Runbook section 9 requires three layers on every paid pod. ``ops/wrapper.py`` has two
of them -- a deadline and an idle watchdog -- and both run *inside* the pod. The
failure they cannot cover is the pod that has wedged badly enough to stop running its
own wrapper, or whose wrapper was never started because the image failed to boot. That
pod bills happily and tells nobody.

So this one runs on the PC and asks the provider directly, which makes its own failure
modes part of the design:

* **An API error must not stop the loop.** A blip that killed the watchdog would leave
  whatever is running unwatched, which is worse than a missed pass. Errors are logged
  and the loop continues.
* **A heartbeat is written every pass, including when nothing is running.** "No pods"
  and "the watchdog is dead" must never look the same in the log, because
  ``/session-start`` reads this file to decide whether the layer was actually up.
* **An unreadable burn tracker does not trigger a mass teardown.** If the file will not
  parse we do not know what is authorised, and destroying everything on that basis
  would turn a typo into a lost training run. It is logged as an error and nothing is
  killed.
* **A failed teardown is the loudest thing here**, because it is the one case where we
  believe a pod is gone and it is still billing.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ops.burn import read_burn_tracker
from ops.guards import pods_to_destroy
from ops.registry import REGISTRY_PATH, forget_pod, pod_states, read_registry

__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "LOG_PATH",
    "MAX_HEARTBEAT_AGE_MINUTES",
    "WatchResult",
    "heartbeat_line",
    "is_fresh",
    "last_heartbeat",
    "main",
    "watch_once",
]

LOG_PATH = Path(r"E:\Geetha\logs\gpu_watch.log")
BURN_PATH = Path(r"E:\Geetha\manifests\burn_tracker.csv")

# Two minutes bounds the overshoot on a 4090 at $0.34/h to about INR 1. Shorter would
# add API calls for nothing; much longer and the deadline stops meaning much.
DEFAULT_INTERVAL_SECONDS = 120

# What /session-start checks: the layer counts as up only if it reported recently.
MAX_HEARTBEAT_AGE_MINUTES = 15

INR_PER_USD = 88.0

_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)")


@dataclass(frozen=True)
class WatchResult:
    live: int
    doomed: int
    destroyed: tuple[str, ...]
    error: str | None


def heartbeat_line(now: datetime, result: WatchResult) -> str:
    parts = [
        now.isoformat(timespec="seconds"),
        f"live={result.live}",
        f"doomed={result.doomed}",
    ]
    if result.destroyed:
        parts.append("destroyed=" + ",".join(result.destroyed))
    if result.error:
        parts.append(f"ERROR {result.error}")
    return "  ".join(parts)


def last_heartbeat(path: str | Path = LOG_PATH) -> datetime | None:
    """Timestamp of the most recent readable heartbeat, or None.

    Scans backwards so that a line truncated by a power cut costs one pass of history
    rather than making the whole log unreadable.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        match = _TIMESTAMP.match(line.strip())
        if not match:
            continue
        try:
            stamp = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
        except ValueError:
            continue
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return None


def is_fresh(
    path: str | Path = LOG_PATH,
    *,
    now: datetime | None = None,
    max_age_minutes: int = MAX_HEARTBEAT_AGE_MINUTES,
) -> bool:
    """Whether the watchdog has reported recently enough to count as running."""
    stamp = last_heartbeat(path)
    if stamp is None:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - stamp).total_seconds() <= max_age_minutes * 60


def watch_once(
    client,
    *,
    now: datetime | None = None,
    registry_path: str | Path = REGISTRY_PATH,
    burn_path: str | Path = BURN_PATH,
) -> WatchResult:
    """One pass: look at every live pod, destroy the ones that must not keep running."""
    now = now or datetime.now(timezone.utc)

    try:
        pods = client.live_pods()
    except Exception as error:
        return WatchResult(live=0, doomed=0, destroyed=(), error=f"live_pods failed: {error}")

    if not pods:
        return WatchResult(live=0, doomed=0, destroyed=(), error=None)

    try:
        burn_rows = read_burn_tracker(burn_path)
    except (OSError, ValueError) as error:
        # We do not know what is authorised, so we destroy nothing.
        return WatchResult(
            live=len(pods), doomed=0, destroyed=(),
            error=f"burn tracker unreadable, destroyed nothing: {error}",
        )

    registry = read_registry(registry_path)
    states = pod_states(pods, registry, inr_per_usd=INR_PER_USD)
    doomed = pods_to_destroy(states, burn_rows, now=now)

    destroyed: list[str] = []
    problems: list[str] = []
    for state, reason in doomed:
        try:
            client.terminate_pod(state.pod_id)
        except Exception as error:
            problems.append(f"{state.pod_id} STILL ALIVE AND BILLING: {error}")
            continue
        forget_pod(state.pod_id, path=registry_path)
        destroyed.append(state.pod_id)

    return WatchResult(
        live=len(pods),
        doomed=len(doomed),
        destroyed=tuple(destroyed),
        error="; ".join(problems) if problems else None,
    )


def _log(line: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="watchdog", description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                        help=f"seconds between passes (default {DEFAULT_INTERVAL_SECONDS})")
    parser.add_argument("--log", default=str(LOG_PATH))
    args = parser.parse_args(argv)

    from ops.runpod_client import RunPodClient
    from ops.secrets import load_secrets

    log_path = Path(args.log)
    client = RunPodClient(load_secrets())

    while True:
        now = datetime.now(timezone.utc)
        result = watch_once(client, now=now)
        line = heartbeat_line(now, result)
        _log(line, log_path)
        print(line, flush=True)
        if result.error:
            print(result.error, file=sys.stderr, flush=True)
        if args.once:
            return 2 if result.error else 0
        time.sleep(max(10, args.interval))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
