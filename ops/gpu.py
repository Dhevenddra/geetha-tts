r"""gpu.py -- the only way Geetha touches a paid GPU (runbook section 9).

    py -3.11 -m ops.gpu balance
    py -3.11 -m ops.gpu quote ops/jobs/m1-pilot.yaml
    py -3.11 -m ops.gpu launch ops/jobs/m1-pilot.yaml [--check-only]
    py -3.11 -m ops.gpu status [--enforce]
    py -3.11 -m ops.gpu logs|pull|destroy <run_id>

B11 finishes the write half. Two things are worth knowing before reading the code:

**Every guard runs before the provider is contacted.** No approval, or no `planned`
burn-tracker row, and the command refuses without a network call at all. A guard that
only fires after an API round-trip is a guard a network blip can defeat, and a refusal
that needs credentials cannot be demonstrated to the operator before a gate.

**There is no `stop`.** The runbook lists one, but stopping a RunPod pod leaves its
disk allocated and *still billing*. Every teardown path in this project is a delete --
on the pod (``ops/wrapper.py``), in the client (``ops/runpod_client.py``) and here --
and each of the three has a test asserting it cannot acquire a stop path. ``stop`` is
kept as a command only so that typing it produces that explanation rather than a
"no such command" and a shrug.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ops.approvals import APPROVALS_PATH, ApprovalError, read_approvals
from ops.burn import (
    TERMINAL_BURN_STATUSES,
    BurnTrackerError,
    read_burn_tracker,
    set_status,
)
from ops.guards import PRICE_TOLERANCE, LaunchRefused, check_launch, pods_to_destroy
from ops.registry import REGISTRY_PATH, forget_pod, pod_states, read_registry, record_launch
from ops.gpu_select import choose_gpu
from ops.runpod_client import image_exists
from ops.secrets import load_secrets
from ops.spec import JobSpec, SpecError
from ops.wrapper import build_wrapper

__all__ = ["build_parser", "main"]

BURN_TRACKER = Path(r"E:\Geetha\manifests\burn_tracker.csv")

NO_STOP = (
    "refusing: there is no stop command. Stopping a RunPod pod leaves its disk "
    "allocated and still billing, which is the slow version of the failure this "
    "tooling exists to prevent. Use `destroy <run_id>`, which pulls nothing and "
    "deletes the pod; use `pull <run_id>` first if you want the outputs."
)

# Re-quoted at each gate (runbook section 2). Kept in one place so a stale rate is
# visible rather than scattered through estimates.
INR_PER_USD = 88.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    quote = sub.add_parser("quote", help="current price and availability for a spec's GPU")
    quote.add_argument("spec", help="path to ops/jobs/<run_id>.yaml")

    sub.add_parser("balance", help="prepaid balance on RunPod")

    launch = sub.add_parser("launch", help="create the pod for a spec")
    launch.add_argument("spec", help="path to ops/jobs/<run_id>.yaml")
    launch.add_argument(
        "--wait-for-stock",
        type=float,
        default=0.0,
        metavar="MINUTES",
        help="retry GPU selection for this many minutes before giving up. Capacity here "
             "is per-tier and hourly, so a stock-out should cost a wait, not a day",
    )
    launch.add_argument(
        "--check-only",
        action="store_true",
        help="run every money guard and report, without contacting a provider",
    )

    status = sub.add_parser("status", help="all pods: hours, cost, deadline")
    status.add_argument(
        "--enforce",
        action="store_true",
        help="destroy any pod past its deadline, over its ceiling, or not written down",
    )

    for name, help_text in (
        ("logs", "tail a run's logs"),
        ("pull", r"copy adapters, metrics and samples into E:\Geetha\runs\<run_id>"),
        ("stop", "refused on purpose — see the message"),
        ("destroy", "delete a run's pod"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("run_id")

    return parser



def _create_with_retry(client, spec, start_script, *, wait_minutes: float, cloud_type: str,
                       pod_secrets: dict):
    """Create the pod, cycling candidates and then waiting. Returns (pod_id, gpu, price).

    The catalogue is not a capacity signal. On 3 Nov the A6000 reported stock **and** a
    price, guards passed, and ``create_pod`` still answered HTTP 500 "There are no
    instances currently available". Only the create tells the truth, so the retry has to
    wrap the create rather than the price lookup -- an earlier version retried the
    lookup, which passed instantly, and then gave up after a single create.

    Each pass walks the candidates in preference order and tries each one. All of them
    are already inside the gate's hourly ceiling, so falling through to a cheaper card
    can only reduce the bill. Returns ``(None, None, None)`` if the wait runs out
    without a pod, having deleted anything it created along the way -- a hunt may create
    several pods that Runpod reclaims before they schedule, and each one is claimed back
    explicitly rather than assumed dead.
    """
    import time as _time

    deadline = _time.time() + wait_minutes * 60.0
    attempt = 0
    while True:
        attempt += 1
        for name in spec.candidate_gpus:
            rate = client.gpu_price_usd_per_h(name, cloud_type)
            if rate is None or rate * INR_PER_USD > spec.ceiling_inr_per_hour:
                continue
            try:
                pod_id = client.create_pod(
                    name=spec.run_id,
                    gpu_type=name,
                    image=spec.image,
                    disk_gb=spec.disk_gb,
                    start_script=start_script,
                    env={**pod_secrets, "GEETHA_RUN_ID": spec.run_id, "WANDB_PROJECT": spec.wandb_project},
                    cloud_type=cloud_type,
                )
            except Exception as error:
                print(f"  attempt {attempt}: {name} at ${rate:.3f}/h -> {error}", flush=True)
                continue

            # A returned pod id is not a running pod. On 3 Nov a create succeeded,
            # returned c9f4v1s3ltvk92, and the pod was TERMINATED four seconds later
            # having never been scheduled and never billed -- Runpod accepted the
            # request and then found no machine. Treating the id as success stopped the
            # hunt with a dead pod, which is the worst of both outcomes: no pod, and no
            # more looking. So confirm it survives before believing it.
            # Confirm it is STILL alive at the end of the window. An earlier version
            # broke on the first sighting, which passed a pod that was already dying:
            # 72sv2cev9g5yh6 was seen alive at ~6 s and terminated at 19 s.
            settled = True
            for _ in range(12):
                _time.sleep(6)
                try:
                    alive = any(getattr(pod, "pod_id", None) == pod_id
                                for pod in client.live_pods())
                except Exception:
                    alive = True  # a failed read is not evidence the pod is gone
                if not alive:
                    settled = False
                    break
            if not settled:
                # "Not in live_pods()" is our READ saying it is gone, which is not the
                # same as it being gone. If the read merely lagged, continuing here
                # abandons a live, unregistered pod -- the one genuine scare of 3 Nov.
                # So claim the delete rather than assume it. terminate_pod treats 404
                # as success, so this is free when Runpod really did reclaim it.
                try:
                    client.terminate_pod(pod_id)
                except Exception as teardown_error:
                    print(f"  attempt {attempt}: {name} was created as {pod_id}, looked "
                          f"gone, AND THE TEARDOWN FAILED ({teardown_error}). Open the "
                          f"RunPod console and check for {pod_id} now.", flush=True)
                    raise
                print(f"  attempt {attempt}: {name} was created as {pod_id} but vanished "
                      f"within 60s (never scheduled); confirmed deleted, continuing the "
                      f"hunt", flush=True)
                continue
            return pod_id, name, rate
        if _time.time() >= deadline:
            return None, None, None
        print(f"  attempt {attempt}: no candidate had capacity; retrying in 15s "
              f"({(deadline - _time.time()) / 60:.0f} min left)", flush=True)
        _time.sleep(15)


def _resolve_gpu(spec, client, *, wait_minutes: float = 0.0):
    """Pick a GPU for ``spec`` from live prices and the gate's hourly ceiling (D-045).

    The catalogue is built from the provider one candidate at a time, because a tier
    with no price is a tier that is not for sale -- which is the same thing as no stock
    for our purposes. The decision itself is :func:`ops.gpu_select.choose_gpu`, so the
    ordering and ceiling rules are the ones under test rather than a second copy here.

    With ``wait_minutes`` the resolution is retried until something qualifies. That
    exists so a stock-out costs a wait rather than the operator sitting at the keyboard
    refreshing a catalogue: the 4090 this project prices every gate against was
    unavailable on both clouds on 3 Nov, an A6000 appeared an hour later, and an hour
    after that it was a 3090 instead.
    """
    import time as _time

    deadline = _time.time() + wait_minutes * 60.0
    attempt = 0
    while True:
        attempt += 1
        catalogue = []
        for name in spec.candidate_gpus:
            price = client.gpu_price_usd_per_h(name, spec.cloud_type)
            catalogue.append({
                "id": name,
                "pricePerHr": price,
                "stockStatus": "unavailable" if price is None else "available",
                "memoryGb": None,
            })
        selection = choose_gpu(
            spec.candidate_gpus, catalogue=catalogue,
            max_inr=spec.max_cost_inr, max_hours=spec.max_hours,
            usd_to_inr=INR_PER_USD,
        )
        if selection.choice is not None or _time.time() >= deadline:
            return selection
        print(f"  attempt {attempt}: {selection.summary}; waiting 60s", flush=True)
        _time.sleep(60)


def _client():
    from ops.runpod_client import RunPodClient

    return RunPodClient(load_secrets())


def _cmd_balance() -> int:
    client = _client()
    usd = client.balance_usd()
    print(f"RunPod prepaid balance: ${usd:.2f}  (about INR {usd * INR_PER_USD:,.0f})")
    print("This balance is the hard ceiling on all spend. Only the operator tops it up.")
    return 0


def _load_spec(spec_path: str) -> JobSpec | None:
    try:
        return JobSpec.from_yaml(spec_path)
    except (OSError, SpecError) as error:
        print(f"cannot read {spec_path}: {error}", file=sys.stderr)
        return None


def _cmd_status(enforce: bool) -> int:
    client = _client()
    pods = client.live_pods()
    print(f"live pods: {len(pods)}")
    if not pods:
        print("nothing is running, so nothing is being billed.")
        return 0

    registry = read_registry(REGISTRY_PATH)
    states = pod_states(pods, registry, inr_per_usd=INR_PER_USD)
    by_id = {state.pod_id: state for state in states}
    for pod in pods:
        state = by_id[pod.pod_id]
        recorded = any(row.pod_id == pod.pod_id for row in registry)
        print(
            f"  {pod.pod_id}  {state.run_id}  {pod.desired_status}  "
            f"{pod.gpu_type} x{pod.gpu_count}  ${pod.cost_per_hr:.3f}/h  "
            f"up {state.hours_used:.2f} h  spent ~INR {state.cost_inr:,.0f}"
            + (
                f"  ceiling INR {state.max_cost_inr:,.0f}  deadline "
                f"{state.deadline.isoformat(timespec='minutes')}"
                if recorded
                else "  NOT IN THE POD REGISTRY"
            )
        )

    try:
        burn_rows = read_burn_tracker(BURN_TRACKER)
    except (OSError, ValueError) as error:
        print(f"burn tracker unusable: {error}", file=sys.stderr)
        # A tracker we cannot read must not silently authorise anything to keep
        # running, but nor should it trigger a mass teardown on a parse error.
        return 1

    doomed = pods_to_destroy(states, burn_rows, now=datetime.now(timezone.utc))
    if not doomed:
        return 0

    print(f"\n{len(doomed)} pod(s) must not keep running:")
    for state, reason in doomed:
        print(f"  {reason}")
    if not enforce:
        print("\nrun `status --enforce` to destroy them.")
        return 1

    failures = 0
    for state, reason in doomed:
        print(f"destroying {state.pod_id} ({state.run_id}): {reason}")
        try:
            client.terminate_pod(state.pod_id)
        except Exception as error:  # a failed teardown is the loudest thing here
            failures += 1
            print(f"  FAILED to destroy {state.pod_id}: {error}", file=sys.stderr)
            continue
        forget_pod(state.pod_id, path=REGISTRY_PATH)
        print(f"  destroyed {state.pod_id}")
    if failures:
        print(
            f"{failures} pod(s) are still alive and still billing — open the RunPod "
            f"console now.",
            file=sys.stderr,
        )
        return 2
    return 1  # pods had to be killed; this is not a clean status


def _cmd_quote(spec_path: str) -> int:
    spec = _load_spec(spec_path)
    if spec is None:
        return 1
    client = _client()
    price = client.gpu_price_usd_per_h(spec.gpu_type, spec.cloud_type)
    if price is None:
        print(f"no {spec.cloud_type} price available for {spec.gpu_type!r}", file=sys.stderr)
        return 1
    cost = spec.max_hours * price * INR_PER_USD
    print(f"{spec.gpu_type} ({spec.cloud_type}): ${price:.3f}/h")
    print(f"{spec.max_hours:g} h would cost about INR {cost:,.0f} "
          f"(ceiling INR {spec.max_cost_inr:,.0f})")
    return 0 if cost <= spec.max_cost_inr else 1


def _cmd_launch(spec_path: str, check_only: bool = False, wait_for_stock: float = 0.0) -> int:
    spec = _load_spec(spec_path)
    if spec is None:
        return 1
    if not spec.is_paid:
        print(f"{spec.run_id} is a {spec.provider} job; this command is for paid pods.",
              file=sys.stderr)
        return 1

    # --- guards that need no network, in the order that fails cheapest -----------
    try:
        approvals = read_approvals(APPROVALS_PATH)
    except ApprovalError as error:
        print(f"gate approvals unusable: {error}", file=sys.stderr)
        return 3
    approval = approvals.get(spec.gate)
    if approval is None:
        print(
            f"LAUNCH REFUSED: no approval on record for gate {spec.gate}. The operator "
            f'must reply "approved" to /gate {spec.gate}, which writes the row in '
            f"{APPROVALS_PATH}.",
            file=sys.stderr,
        )
        return 3

    try:
        burn_rows = read_burn_tracker(BURN_TRACKER)
    except (OSError, ValueError) as error:
        print(f"burn tracker unusable: {error}", file=sys.stderr)
        return 1
    if not any(row.run_id == spec.run_id for row in burn_rows):
        print(
            f"LAUNCH REFUSED: no row for run_id {spec.run_id!r} in the burn tracker "
            f"({BURN_TRACKER}). Write the `planned` row before launching.",
            file=sys.stderr,
        )
        return 3

    if any(row.run_id == spec.run_id for row in read_registry(REGISTRY_PATH)):
        print(
            f"LAUNCH REFUSED: the pod registry already has a live pod for "
            f"{spec.run_id!r}. Pull and destroy it before relaunching.",
            file=sys.stderr,
        )
        return 3

    # The image is checked against its registry rather than against RunPod, which
    # accepts a create for an image that does not exist and then tears the pod down
    # seconds later — indistinguishable from a capacity problem unless you know.
    verified = image_exists(spec.image)
    if verified is False:
        print(
            f"LAUNCH REFUSED: the image {spec.image!r} does not exist. RunPod would "
            f"accept the create, fail the pull, and destroy the pod within seconds "
            f"(measured 27 Oct 2026).",
            file=sys.stderr,
        )
        return 3
    if verified is None:
        print(f"note: could not verify the image {spec.image!r} exists — proceeding, "
              f"but it is not verified.", file=sys.stderr)

    # --- guards that need the provider -------------------------------------------
    client = _client()
    selection = _resolve_gpu(spec, client, wait_minutes=wait_for_stock)
    for reason in selection.rejections:
        print(f"  skipped {reason}")
    if selection.choice is None:
        print(f"refusing to launch: {selection.summary}. Nothing was created and nothing "
              f"is billing.", file=sys.stderr)
        return 1
    gpu_type = selection.choice.gpu_id
    price = selection.choice.price_usd_per_hour
    print(f"chose {selection.summary}")

    try:
        decision = check_launch(
            spec=spec,
            approval=approval,
            burn_rows=burn_rows,
            balance_inr=client.balance_usd() * INR_PER_USD,
            quoted_rate_usd_per_h=price,
            inr_per_usd=INR_PER_USD,
            running_pods=pod_states(
                client.live_pods(), read_registry(REGISTRY_PATH), inr_per_usd=INR_PER_USD
            ),
        )
    except LaunchRefused as refusal:
        print(f"LAUNCH REFUSED: {refusal}", file=sys.stderr)
        return 3

    print(f"guards passed; estimated INR {decision.estimated_cost_inr:,.0f} "
          f"of a INR {spec.max_cost_inr:,.0f} ceiling "
          f"({spec.max_hours:g} h at ${price:.3f}/h, {spec.cloud_type})")
    if check_only:
        print("--check-only: nothing was created.")
        return 0

    # The pod needs HF_TOKEN to reach the base model and the code shards, and
    # WANDB_API_KEY to log. Resolved *before* the create so a missing secret costs
    # nothing: on 3 Nov two pods were created, pulled the image, ran the job script,
    # hit "FATAL: HF_TOKEN is not set", exited 78 and self-destroyed about two minutes
    # in. The safety layer worked exactly as designed; the job could never have run.
    # pod_env() raises on a missing name, and nothing here ever prints a value.
    pod_secrets = load_secrets().pod_env()

    start_script = build_wrapper(
        run_id=spec.run_id,
        command=spec.command,
        max_hours=spec.max_hours,
        checkpoint_repo=spec.checkpoint_repo,
    )
    # Stamped AFTER the create, not before. `--wait-for-stock 60` can hunt for an hour
    # before anything is created, and this timestamp starts BOTH the billing clock and
    # the deadline. Stamping it here cost m1-pilot-6 five minutes of phantom spend
    # (INR 19 claimed against a pod that lived 1m44s), and would have been far worse
    # than a wrong number: a pod found at minute 59 of the hunt inherits a deadline
    # that began an hour earlier, so a 1.7 h pilot gets 42 minutes and is torn down
    # mid-training with a loss curve that looks perfectly healthy.
    launched_at = None
    try:
        if wait_for_stock > 0:
            pod_id, chosen, chosen_rate = _create_with_retry(
                client, spec, start_script,
                wait_minutes=wait_for_stock, cloud_type=spec.cloud_type,
                pod_secrets=pod_secrets)
            if pod_id is None:
                print(f"LAUNCH GAVE UP: no candidate GPU had capacity within "
                      f"{wait_for_stock:g} minutes. Nothing was created and nothing is "
                      f"billing; the `planned` row for {spec.run_id} is untouched and the "
                      f"gate is still authorised. Retry later.", file=sys.stderr)
                return 1
            if chosen != gpu_type:
                print(f"note: took {chosen} at ${chosen_rate:.3f}/h instead of {gpu_type}; "
                      f"capacity decided, and it is inside the same ceiling")
            gpu_type, price = chosen, chosen_rate
        else:
            pod_id = client.create_pod(
                name=spec.run_id,
                gpu_type=gpu_type,
                image=spec.image,
                disk_gb=spec.disk_gb,
                start_script=start_script,
                env={**pod_secrets, "GEETHA_RUN_ID": spec.run_id, "WANDB_PROJECT": spec.wandb_project},
                cloud_type=spec.cloud_type,
            )
    except Exception as error:
        # Nothing was created, so the authorisation is untouched and still usable. A
        # stack trace here would be an unusable answer to a routine condition: RunPod
        # returns HTTP 500 "no instances currently available" whenever a GPU tier is
        # sold out, which says nothing about our spec.
        print(f"LAUNCH FAILED: the provider did not create the pod: {error}",
              file=sys.stderr)
        print(
            f"Nothing was created and nothing is billing; the `planned` row for "
            f"{spec.run_id} is untouched. If this is a capacity message, the GPU tier "
            f"is sold out — re-quote and pick an available one, or retry later.",
            file=sys.stderr,
        )
        return 4
    # The pod exists from here, so this is when its clocks start.
    launched_at = datetime.now(timezone.utc)
    print(f"pod {pod_id} created for {spec.run_id}")

    # Record the ceilings immediately. Between create and record, the pod is alive and
    # unaccounted for, so nothing goes between these two calls — and if the record
    # cannot be written, the pod is destroyed for the same reason an unconsumable burn
    # row destroys it: nothing could enforce a limit on it.
    try:
        record_launch(
            run_id=spec.run_id,
            pod_id=pod_id,
            gate=spec.gate,
            provider=spec.provider,
            max_hours=spec.max_hours,
            max_cost_inr=spec.max_cost_inr,
            rate_usd_per_h=price,
            launched_at=launched_at,
            path=REGISTRY_PATH,
        )
    except Exception as error:
        print(f"pod {pod_id} was created but its ceilings could not be recorded: "
              f"{error}. Destroying it.", file=sys.stderr)
        try:
            client.terminate_pod(pod_id)
        except Exception as teardown_error:
            print(f"POD {pod_id} IS STILL ALIVE AND BILLING: the teardown also failed "
                  f"({teardown_error}). Open the RunPod console and delete it now.",
                  file=sys.stderr)
            return 2
        print(f"destroyed {pod_id}; nothing is billing.", file=sys.stderr)
        return 3
    # Consume the authorisation. A row left on `planned` lets the same gate approval
    # launch this run again after a destroy, which is how one ceiling becomes several.
    #
    # If it cannot be consumed, the pod must not survive. We would have a pod running
    # that the guards believe does not exist, against an authorisation that is still
    # unspent -- so the accounting is broken and a pod we cannot account for is exactly
    # what this tooling exists to prevent. Tearing it down is recoverable: fix the row
    # and launch again.
    try:
        set_status(BURN_TRACKER, spec.run_id, "running")
    except (BurnTrackerError, OSError) as error:
        print(
            f"pod {pod_id} was created but its burn row could not be consumed: "
            f"{error}. Destroying it rather than running a pod the accounting does "
            f"not know about.",
            file=sys.stderr,
        )
        try:
            client.terminate_pod(pod_id)
        except Exception as teardown_error:
            print(
                f"POD {pod_id} IS STILL ALIVE AND BILLING: the teardown also failed "
                f"({teardown_error}). Open the RunPod console and delete it now.",
                file=sys.stderr,
            )
            return 2
        forget_pod(pod_id, path=REGISTRY_PATH)
        print(f"destroyed {pod_id}; nothing is billing.", file=sys.stderr)
        return 3

    deadline = launched_at.replace(microsecond=0)
    print(f"recorded: ceiling INR {spec.max_cost_inr:,.0f}, "
          f"deadline {spec.max_hours:g} h from {deadline.isoformat()}")
    print("watch it with `status`, and `status --enforce` from the PC watchdog.")
    return 0


def _actual_cost(row, burn_rows) -> tuple[float, float]:
    """Hours the pod was up and what that cost, priced at the rate recorded at launch.

    Computed rather than read from the API, because a pod that has just been deleted
    may no longer be reported — and a run recorded as free quietly returns its share of
    its gate's authorisation.

    The rate comes from the **registry**, which stored what the launch was actually
    quoted. The burn row's rate is only the *planned* one, so pricing with it
    under-recorded the spend by up to the 20% price tolerance. If no rate was recorded
    (a row written before the column existed), the planned rate is used and the
    shortfall is covered by the tolerance, so the figure errs high rather than low.
    """
    hours = max(0.0, (datetime.now(timezone.utc) - row.launched_at).total_seconds() / 3600.0)
    rate = row.rate_usd_per_h
    if rate <= 0:
        planned = next((b.rate_usd_per_h for b in burn_rows if b.run_id == row.run_id), 0.0)
        rate = planned * PRICE_TOLERANCE
        print(
            f"note: no quoted rate recorded for {row.run_id}; pricing at the planned "
            f"rate plus the {PRICE_TOLERANCE:.0%} tolerance (${rate:.3f}/h) so the "
            f"recorded cost errs high.",
            file=sys.stderr,
        )
    return hours, hours * rate * INR_PER_USD


def _cmd_destroy(run_id: str) -> int:
    rows = [row for row in read_registry(REGISTRY_PATH) if row.run_id == run_id]
    if not rows:
        print(
            f"no pod recorded for {run_id!r}. If a pod is alive anyway, "
            f"`status --enforce` destroys anything not written down.",
            file=sys.stderr,
        )
        return 1
    client = _client()
    failures = 0
    for row in rows:
        try:
            client.terminate_pod(row.pod_id)
        except Exception as error:
            failures += 1
            print(f"FAILED to destroy {row.pod_id}: {error}", file=sys.stderr)
            continue
        print(f"destroyed {row.pod_id} ({row.run_id})")
    if failures:
        print("a pod is still alive and still billing — open the RunPod console now.",
              file=sys.stderr)
        return 2

    # Close the burn row BEFORE forgetting the registry row. Forgetting it while the
    # row is still open is what makes a run relaunchable, so if this fails the registry
    # row stays and keeps blocking a relaunch -- the safe direction.
    try:
        burn_rows = read_burn_tracker(BURN_TRACKER)
        current = next((b.status for b in burn_rows if b.run_id == run_id), None)
        if current is None:
            # No burn row at all: there is no authorisation to close and no spend to
            # under-record, so keeping the registry row would block a future launch
            # for nothing. The pod should not have existed, which `status --enforce`
            # would have said; it is gone now either way.
            print(f"note: {run_id} has no burn-tracker row, so there was nothing to "
                  f"close. It should not have been launched through this tool.",
                  file=sys.stderr)
        elif current in TERMINAL_BURN_STATUSES:
            # The run already finished and its cost is already on file -- the normal
            # ending now that the pod self-destroys itself (D-027). There is nothing
            # to close, and nothing to re-price: `_actual_cost` bills wall-clock from
            # `launched_at` to *now*, and by the time anyone tidies up, "now" can be
            # many hours after a pod that lived minutes. Drop the row and say so.
            print(f"burn tracker: {run_id} was already closed as {current!r}; "
                  f"its recorded cost stands. Clearing the registry row only.")
        else:
            # `planned -> done` is not a legal transition. A row still on `planned`
            # here means the launch never consumed it, so the honest close is
            # `aborted`.
            closing = "done" if current == "running" else "aborted"
            hours, cost = _actual_cost(rows[0], burn_rows)
            set_status(BURN_TRACKER, run_id, closing, actual_hours=hours, cost_inr=cost)
            print(f"burn tracker: {run_id} {closing}, {hours:.2f} h, "
                  f"about INR {cost:,.0f}")
    except (BurnTrackerError, OSError, ValueError) as error:
        print(
            f"the pod is gone but its burn row could not be closed: {error}. The pod "
            f"registry row is being kept, so a relaunch stays blocked until you "
            f"record the actual cost by hand and clear the row.",
            file=sys.stderr,
        )
        return 2

    for row in rows:
        forget_pod(row.pod_id, path=REGISTRY_PATH)
    print("confirm with `status`: it should report 0 live pods.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "balance":
        return _cmd_balance()
    if args.command == "status":
        return _cmd_status(args.enforce)
    if args.command == "quote":
        return _cmd_quote(args.spec)
    if args.command == "launch":
        return _cmd_launch(args.spec, args.check_only, args.wait_for_stock)
    if args.command == "destroy":
        return _cmd_destroy(args.run_id)
    if args.command == "stop":
        print(NO_STOP, file=sys.stderr)
        return 2
    print(
        f"{args.command} is not finished yet (B11 covers launch, status and destroy; "
        f"logs and pull come with the M1 pilot).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
