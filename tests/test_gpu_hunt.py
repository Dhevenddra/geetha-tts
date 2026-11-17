"""The stock hunt: what happens to pods Runpod accepts and then reclaims.

`_create_with_retry` is the only code path in the project that creates pods in a loop,
and on 3 Nov 2026 it created fourteen of them. It had no test at all. The audit log is
what exposed the gap: every one of those pods was created, assigned a machineId, and
deleted within 2-19 seconds, and the launcher's own log said "vanished within 60s
(never scheduled); continuing the hunt" -- which is the hunt deciding, from a single
read, that a pod it created no longer exists, and then walking away from it.

That inference is the bug. `live_pods()` not listing a pod is our READ saying it is
gone. If the read lags -- and the whole reason this project has a registry is that the
Runpod APIs lie -- the hunt abandons a live, unregistered, billing pod and then creates
another one. Fourteen times.

So the property under test is not "the hunt finds a pod". It is: **the hunt never ends
a pass with a pod it created still unaccounted for.**
"""

import types

import pytest

from ops import gpu


class HuntClient:
    """A client whose pods are accepted and then never scheduled.

    `live_pods` always answers empty, which is exactly what the launcher saw on 3 Nov.
    """

    def __init__(self, price=0.59, pod_ids=("pod-a", "pod-b")):
        self._price = price
        self._pod_ids = list(pod_ids)
        self.created = []
        self.terminated = []

    def gpu_price_usd_per_h(self, gpu_type, cloud_type=None):
        return self._price

    def live_pods(self):
        return []

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return self._pod_ids[len(self.created) - 1]

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        return True


def _spec(candidates=("NVIDIA A40",), ceiling=80.0):
    return types.SimpleNamespace(
        run_id="m1-pilot-test",
        image="runpod/pytorch:x",
        disk_gb=40,
        wandb_project="geetha",
        candidate_gpus=list(candidates),
        ceiling_inr_per_hour=ceiling,
    )


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """The hunt sleeps 6 s twelve times per pod. Tests must not."""
    import time

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


def test_a_pod_that_vanishes_is_deleted_not_abandoned():
    """The 3 Nov failure, reproduced.

    Runpod accepts the create and the pod never appears alive. The hunt must claim the
    delete rather than assume it -- otherwise a lagging read leaves a billing pod that
    no registry row, no watchdog and no burn ceiling knows about.
    """
    client = HuntClient()

    pod_id, gpu_name, rate = gpu._create_with_retry(
        client, _spec(), "echo hi",
        wait_minutes=0.0, cloud_type="SECURE", pod_secrets={},
    )

    assert (pod_id, gpu_name, rate) == (None, None, None)
    assert client.created, "the hunt should have attempted a create"
    assert client.terminated == ["pod-a"], (
        "the hunt created pod-a, decided it had vanished, and walked away without "
        "deleting it -- this is how a live pod becomes untracked"
    )


def test_every_pod_created_during_a_hunt_is_accounted_for():
    """Across several candidates, nothing is left behind.

    The real hunt walked four candidates per pass for ten passes. The invariant has to
    hold for the whole hunt, not just the first create.
    """
    client = HuntClient(pod_ids=("pod-a", "pod-b", "pod-c"))

    gpu._create_with_retry(
        client, _spec(candidates=("NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40S")),
        "echo hi", wait_minutes=0.0, cloud_type="SECURE", pod_secrets={},
    )

    created_ids = ["pod-a", "pod-b", "pod-c"][: len(client.created)]
    assert client.terminated == created_ids, (
        f"created {created_ids} but only deleted {client.terminated}; "
        f"{set(created_ids) - set(client.terminated)} were abandoned"
    )


def test_a_failed_teardown_is_raised_not_swallowed():
    """A pod we cannot delete is the loudest thing in this project.

    If the delete fails, continuing the hunt would create a second pod on top of a
    first one that may still be billing. Stop instead.
    """
    client = HuntClient()
    client.terminate_pod = lambda pod_id: (_ for _ in ()).throw(RuntimeError("HTTP 500"))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        gpu._create_with_retry(
            client, _spec(), "echo hi",
            wait_minutes=0.0, cloud_type="SECURE", pod_secrets={},
        )


def test_a_candidate_over_the_ceiling_is_never_created():
    """The price guard sits before the create, so an expensive card costs nothing."""
    client = HuntClient(price=10.0)  # ~INR 880/h, far over any gate

    pod_id, _, _ = gpu._create_with_retry(
        client, _spec(ceiling=80.0), "echo hi",
        wait_minutes=0.0, cloud_type="SECURE", pod_secrets={},
    )

    assert pod_id is None
    assert client.created == [], "a card over the hourly ceiling must not be created"


def test_the_clock_starts_when_the_pod_is_created_not_when_the_hunt_starts():
    """`launched_at` must be stamped after the create, not before the hunt.

    It starts both the billing clock and the deadline. Stamped before a
    `--wait-for-stock 60` hunt, a pod found at minute 59 inherits a deadline that began
    an hour earlier: a 1.7 h pilot would get 42 minutes and be destroyed mid-training
    with a healthy-looking loss curve. m1-pilot-6 showed the cheap half of this -- INR 19
    of phantom spend claimed against a pod that lived 1m44s.

    This is a source-ordering assertion rather than a behavioural one. Driving
    `_cmd_launch` end to end needs an approvals file, a burn tracker, a registry and a
    spec, and the thing that actually broke is one statement sitting on the wrong side
    of one call -- which is exactly what this reads. It is a weaker test than a
    behavioural one and is marked as such, not dressed up as more.
    """
    import inspect

    from ops import gpu as gpu_mod

    source = inspect.getsource(gpu_mod._cmd_launch)
    stamp = source.index("launched_at = datetime.now(timezone.utc)")
    hunt = source.index("_create_with_retry")
    assert stamp > hunt, (
        "launched_at is stamped before the stock hunt runs; a long --wait-for-stock "
        "would then start the deadline and the billing clock in the past"
    )
