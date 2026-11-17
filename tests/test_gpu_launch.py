"""Tests for the money-spending commands of ops.gpu, finished in B11.

Everything here is about the order in which refusals happen. A guard that only fires
after a provider call is a guard that can be defeated by a network blip, and a refusal
that needs credentials cannot be demonstrated to the operator before a gate.
"""
import csv
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from ops import gpu
from ops.approvals import APPROVAL_COLUMNS
from ops.burn import BURN_COLUMNS
from ops.registry import REGISTRY_COLUMNS, read_registry, record_launch
from ops.runpod_client import RunPodPod

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)

SPEC = {
    "run_id": "m1-pilot",
    "gate": "M1",
    "provider": "runpod",
    "gpu_type": "NVIDIA GeForce RTX 4090",
    "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "disk_gb": 60,
    "max_hours": 2.0,
    "max_cost_inr": 300.0,
    "command": "python train.py",
    "data_repos": ["dhevenddra/geetha-stage-a"],
    "checkpoint_repo": "dhevenddra/geetha-ckpt",
    "wandb_project": "geetha",
}


class FakeClient:
    def __init__(self, balance_usd=20.0, price=0.34, pods=(), pod_id="pod-new"):
        self._balance = balance_usd
        self._price = price
        self._pods = list(pods)
        self._pod_id = pod_id
        self.created = []
        self.terminated = []

    def balance_usd(self):
        return self._balance

    def gpu_price_usd_per_h(self, gpu_type, cloud_type=None):
        return self._price

    def live_pods(self):
        return list(self._pods)

    def create_pod(self, **kwargs):
        self.created.append(kwargs)
        return self._pod_id

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        return True


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A spec, a burn tracker, an approvals file and a registry, all redirected."""
    spec_path = tmp_path / "m1-pilot.yaml"
    spec_path.write_text(yaml.safe_dump(SPEC), encoding="utf-8")

    burn = tmp_path / "burn_tracker.csv"
    with burn.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BURN_COLUMNS)
    approvals = tmp_path / "gate_approvals.csv"
    registry = tmp_path / "pod_registry.csv"

    monkeypatch.setattr(gpu, "BURN_TRACKER", burn)
    monkeypatch.setattr(gpu, "APPROVALS_PATH", approvals)
    monkeypatch.setattr(gpu, "REGISTRY_PATH", registry)
    return {"spec": spec_path, "burn": burn, "approvals": approvals, "registry": registry}


def _approve(workspace, gate="M1", max_inr=300):
    with workspace["approvals"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(APPROVAL_COLUMNS)
        writer.writerow(["2026-11-01", gate, f"{max_inr}", "operator", "test"])


def _plan(workspace, run_id="m1-pilot", gate="M1", hours=2.0, rate=0.34, ceiling=300.0):
    # The ceiling mirrors the job spec: exposure against a gate is counted on ceilings,
    # not on expected cost, so a planned row without one under-reports what it commits.
    with workspace["burn"].open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            "2026-11-01", gate, run_id, "runpod", "RTX 4090", f"{hours}", "",
            f"{rate}", "88", f"{ceiling}", "", "", "planned", "pilot", "",
        ])


def _use(monkeypatch, client):
    monkeypatch.setattr(gpu, "_client", lambda: client)
    return client


# --- stop is gone for good, not pending ------------------------------------------


def test_stop_is_refused_permanently_with_the_billing_reason(capsys, monkeypatch):
    # The runbook lists a stop command, but stopping a RunPod pod leaves its disk
    # allocated and still billing. The refusal must say so, not say "B11", or someone
    # will come back later and implement it.
    code = gpu.main(["stop", "m1-pilot"])
    error = capsys.readouterr().err
    assert code != 0
    assert "billing" in error.lower()
    assert "destroy" in error.lower()
    assert "B11" not in error


# --- launch refuses before touching the network ----------------------------------


def test_launch_without_an_approval_refuses_without_contacting_the_provider(
    workspace, capsys, monkeypatch
):
    def explode():
        raise AssertionError("the provider must not be contacted before the guards run")

    monkeypatch.setattr(gpu, "_client", explode)
    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert "approved" in capsys.readouterr().err


def test_launch_without_a_planned_burn_row_refuses_without_contacting_the_provider(
    workspace, capsys, monkeypatch
):
    _approve(workspace)

    def explode():
        raise AssertionError("the provider must not be contacted before the guards run")

    monkeypatch.setattr(gpu, "_client", explode)
    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert "burn tracker" in capsys.readouterr().err


def test_launch_refuses_a_spec_over_its_gate_authorisation(workspace, capsys, monkeypatch):
    _approve(workspace, max_inr=200)  # operator approved less than the spec asks
    _plan(workspace)
    monkeypatch.setattr(gpu, "_client", lambda: FakeClient())
    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert "authorisation" in capsys.readouterr().err


def test_check_only_passes_the_guards_without_creating_a_pod(workspace, monkeypatch, capsys):
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())
    assert gpu.main(["launch", str(workspace["spec"]), "--check-only"]) == 0
    assert client.created == []
    assert "guards passed" in capsys.readouterr().out


# --- a real launch ----------------------------------------------------------------


def test_launch_creates_the_pod_and_records_its_deadline_and_ceiling(
    workspace, monkeypatch, capsys
):
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())
    assert gpu.main(["launch", str(workspace["spec"])]) == 0

    assert len(client.created) == 1
    rows = read_registry(workspace["registry"])
    assert len(rows) == 1
    assert rows[0].run_id == "m1-pilot"
    assert rows[0].pod_id == "pod-new"
    assert rows[0].max_cost_inr == 300.0


def test_the_created_pod_carries_the_wrapper_with_its_teardown_trap(workspace, monkeypatch):
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())
    gpu.main(["launch", str(workspace["spec"])])

    script = client.created[0]["start_script"]
    assert "trap geetha_self_destroy EXIT" in script
    assert "timeout" in script
    assert "geetha_idle_watchdog" in script
    # Delete, never stop -- on the pod as well as on the PC.
    assert "runpodctl stop" not in script


def test_the_pod_is_named_after_the_run_id(workspace, monkeypatch):
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())
    gpu.main(["launch", str(workspace["spec"])])
    assert client.created[0]["name"] == "m1-pilot"


def test_launch_refuses_when_a_pod_for_that_run_id_is_already_recorded(
    workspace, monkeypatch, capsys
):
    _approve(workspace)
    _plan(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-old", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    client = _use(monkeypatch, FakeClient(
        pods=[RunPodPod("pod-old", "m1-pilot", "RUNNING", "x", 1, 0.34, 60, {})]
    ))
    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert client.created == []


# --- destroy ----------------------------------------------------------------------


def test_destroy_terminates_the_pod_and_forgets_the_row(workspace, monkeypatch):
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    client = _use(monkeypatch, FakeClient())
    assert gpu.main(["destroy", "m1-pilot"]) == 0
    assert client.terminated == ["pod-1"]
    assert read_registry(workspace["registry"]) == []


def test_destroy_of_an_unknown_run_is_an_error_not_a_silent_success(
    workspace, monkeypatch, capsys
):
    client = _use(monkeypatch, FakeClient())
    code = gpu.main(["destroy", "never-launched"])
    assert code != 0
    assert client.terminated == []


# --- the watchdog's enforcement path ---------------------------------------------


def test_enforce_destroys_a_pod_past_its_deadline(workspace, monkeypatch, capsys):
    _plan(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=1.0, max_cost_inr=300.0,
                  launched_at=datetime.now(timezone.utc) - timedelta(hours=3),
                  path=workspace["registry"])
    client = _use(monkeypatch, FakeClient(
        pods=[RunPodPod("pod-1", "m1-pilot", "RUNNING", "x", 1, 0.34, 10800, {})]
    ))
    code = gpu.main(["status", "--enforce"])
    assert client.terminated == ["pod-1"]
    assert code != 0  # a pod that had to be killed is not a clean status
    assert "deadline" in capsys.readouterr().out


def test_enforce_destroys_a_pod_nobody_wrote_down(workspace, monkeypatch):
    # No registry row, no burn row: either a launch that bypassed the guards or a
    # leftover from a crashed session. Both cost money.
    client = _use(monkeypatch, FakeClient(
        pods=[RunPodPod("stray", "mystery", "RUNNING", "x", 1, 1.19, 7200, {})]
    ))
    gpu.main(["status", "--enforce"])
    assert client.terminated == ["stray"]


def test_enforce_leaves_a_pod_inside_its_limits_alone(workspace, monkeypatch):
    _plan(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=4.0, max_cost_inr=300.0, launched_at=datetime.now(timezone.utc),
                  path=workspace["registry"])
    client = _use(monkeypatch, FakeClient(
        pods=[RunPodPod("pod-1", "m1-pilot", "RUNNING", "x", 1, 0.34, 600, {})]
    ))
    assert gpu.main(["status", "--enforce"]) == 0
    assert client.terminated == []


def test_status_without_enforce_never_destroys_anything(workspace, monkeypatch):
    _plan(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=1.0, max_cost_inr=300.0,
                  launched_at=datetime.now(timezone.utc) - timedelta(hours=5),
                  path=workspace["registry"])
    client = _use(monkeypatch, FakeClient(
        pods=[RunPodPod("pod-1", "m1-pilot", "RUNNING", "x", 1, 0.34, 18000, {})]
    ))
    gpu.main(["status"])
    assert client.terminated == []


# --- the planned row is consumed, so one approval is one launch ------------------
#
# Flagged by security review, 27 Oct: destroy forgot the registry row but left the burn
# row on `planned`, so the same gate approval permitted unlimited relaunches.


def test_launching_moves_the_burn_row_from_planned_to_running(workspace, monkeypatch):
    from ops.burn import read_burn_tracker

    _approve(workspace)
    _plan(workspace)
    _use(monkeypatch, FakeClient())
    gpu.main(["launch", str(workspace["spec"])])
    assert read_burn_tracker(workspace["burn"])[0].status == "running"


def test_a_destroyed_run_cannot_be_relaunched_on_the_same_approval(
    workspace, monkeypatch, capsys
):
    from ops.burn import read_burn_tracker

    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())

    assert gpu.main(["launch", str(workspace["spec"])]) == 0
    assert gpu.main(["destroy", "m1-pilot"]) == 0
    assert read_burn_tracker(workspace["burn"])[0].status == "done"

    # The registry row is gone, so the registry no longer blocks it. The consumed
    # burn row must.
    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert len(client.created) == 1
    assert "'done'" in capsys.readouterr().err


def test_destroy_records_what_the_run_actually_cost(workspace, monkeypatch):
    from ops.burn import read_burn_tracker

    _approve(workspace)
    _plan(workspace)
    _use(monkeypatch, FakeClient())
    gpu.main(["launch", str(workspace["spec"])])
    gpu.main(["destroy", "m1-pilot"])
    row = read_burn_tracker(workspace["burn"])[0]
    # Launched and destroyed within the same test, so the cost is ~0 but must be
    # *recorded*: a run with no cost on file reads as free and silently returns its
    # share of the gate's authorisation.
    assert row.status == "done"
    assert "actual_hours" in workspace["burn"].read_text(encoding="utf-8")


def test_over_planning_a_gate_is_caught_at_the_first_launch_not_the_second(
    workspace, monkeypatch, capsys
):
    # Two runs, each authorised to INR 300, under an M1 approved for INR 300. Counting
    # exposure on *ceilings* catches this before anything is created, rather than
    # letting the first run through and discovering the problem on the second.
    _approve(workspace, max_inr=300)
    _plan(workspace, run_id="m1-pilot", ceiling=300.0)
    _plan(workspace, run_id="m1-pilot-again", ceiling=300.0)

    client = _use(monkeypatch, FakeClient())
    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert client.created == []
    assert "already committed" in capsys.readouterr().err


def test_runs_that_fit_the_gate_together_both_launch(workspace, monkeypatch):
    # M4 is INR 1,300 (D-013); two runs of INR 600 fit, so neither is refused.
    import yaml

    _approve(workspace, gate="M4", max_inr=1300)
    _plan(workspace, run_id="m4-asr-one", gate="M4", ceiling=600.0)
    _plan(workspace, run_id="m4-asr-two", gate="M4", ceiling=600.0)

    client = _use(monkeypatch, FakeClient())
    for index, run_id in enumerate(("m4-asr-one", "m4-asr-two")):
        spec = SPEC | {"run_id": run_id, "gate": "M4", "max_cost_inr": 600.0}
        path = workspace["spec"].parent / f"{run_id}.yaml"
        path.write_text(yaml.safe_dump(spec), encoding="utf-8")
        client._pod_id = f"pod-{index}"
        assert gpu.main(["launch", str(path)]) == 0
    assert len(client.created) == 2


def test_a_third_run_that_overruns_the_gate_is_refused(workspace, monkeypatch, capsys):
    import yaml

    _approve(workspace, gate="M4", max_inr=1300)
    for run_id in ("m4-asr-one", "m4-asr-two", "m4-asr-three"):
        _plan(workspace, run_id=run_id, gate="M4", ceiling=600.0)

    client = _use(monkeypatch, FakeClient())
    spec = SPEC | {"run_id": "m4-asr-one", "gate": "M4", "max_cost_inr": 600.0}
    path = workspace["spec"].parent / "m4-asr-one.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    # 600 (this) + 600 + 600 (the other two planned) = 1,800 against 1,300.
    assert gpu.main(["launch", str(path)]) != 0
    assert client.created == []
    assert "1,800" in capsys.readouterr().err


# --- the authorisation must actually be consumed, or the pod must not survive -----
#
# Second security review, 27 Oct. If set_status(..., "running") failed after the pod
# was created, launch printed a WARNING and returned 0: the pod ran, the row stayed
# `planned`, and the run was relaunchable. Destroy then tried planned -> done, which is
# not an allowed transition, so it warned and forgot the registry row anyway, leaving
# the row stuck on `planned` for good.


class BrokenTracker(FakeClient):
    """A client that works, paired with a tracker that cannot be written."""


def test_a_pod_is_destroyed_if_its_authorisation_cannot_be_consumed(
    workspace, monkeypatch, capsys
):
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())

    from ops import burn

    def refuse(*args, **kwargs):
        raise burn.BurnTrackerError("disk full")

    monkeypatch.setattr(gpu, "set_status", refuse)

    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    # The pod was created, then torn down: we cannot account for it, so it must not run.
    assert len(client.created) == 1
    assert client.terminated == ["pod-new"]
    assert read_registry(workspace["registry"]) == []
    assert "could not be" in capsys.readouterr().err


def test_if_that_teardown_also_fails_the_message_is_unmissable(
    workspace, monkeypatch, capsys
):
    _approve(workspace)
    _plan(workspace)

    class Stubborn(FakeClient):
        def terminate_pod(self, pod_id):
            raise RuntimeError("HTTP 500")

    _use(monkeypatch, Stubborn())

    from ops import burn

    monkeypatch.setattr(gpu, "set_status",
                        lambda *a, **k: (_ for _ in ()).throw(burn.BurnTrackerError("x")))

    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    error = capsys.readouterr().err
    assert "STILL ALIVE" in error
    assert "console" in error.lower()


def test_destroy_closes_a_row_still_on_planned_rather_than_leaving_it_stuck(
    workspace, monkeypatch
):
    from ops.burn import read_burn_tracker

    # Simulate the drifted state the old code could leave behind: pod recorded, burn
    # row never advanced past `planned`.
    _plan(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    _use(monkeypatch, FakeClient())

    assert gpu.main(["destroy", "m1-pilot"]) == 0
    row = read_burn_tracker(workspace["burn"])[0]
    # planned -> done is not a legal transition, so the recovery path aborts it.
    assert row.status == "aborted"
    assert read_registry(workspace["registry"]) == []


def test_destroy_keeps_the_registry_row_if_the_burn_row_cannot_be_closed(
    workspace, monkeypatch, capsys
):
    # Forgetting the registry row while the burn row is unclosed is what makes a run
    # relaunchable. Keeping it blocks relaunch, which is the safe direction.
    _plan(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    _use(monkeypatch, FakeClient())

    from ops import burn

    monkeypatch.setattr(gpu, "set_status",
                        lambda *a, **k: (_ for _ in ()).throw(burn.BurnTrackerError("x")))

    code = gpu.main(["destroy", "m1-pilot"])
    assert code != 0
    assert read_registry(workspace["registry"]) != []
    assert "by hand" in capsys.readouterr().err


# --- the provider refusing to create a pod is not a crash ------------------------
#
# Hit for real on 27 Oct: RunPod returned HTTP 500 "There are no instances currently
# available" and the RunPodError propagated as a traceback. State stayed correct (no
# pod, row still `planned`), but a stack trace is not a usable answer, and a launch
# path that can crash is one whose exit code cannot be trusted by a script.


def test_a_provider_refusing_to_create_is_reported_cleanly(workspace, monkeypatch, capsys):
    from ops.runpod_client import RunPodError
    from ops.burn import read_burn_tracker

    _approve(workspace)
    _plan(workspace)

    class NoCapacity(FakeClient):
        def create_pod(self, **kwargs):
            raise RunPodError(
                'RunPod create pod returned HTTP 500: '
                '{"error":"create pod: There are no instances currently available"}'
            )

    client = _use(monkeypatch, NoCapacity())
    code = gpu.main(["launch", str(workspace["spec"])])

    assert code != 0
    error = capsys.readouterr().err
    assert "no instances" in error.lower()
    assert "Traceback" not in error
    # Nothing was created, so the authorisation must still be intact and reusable.
    assert read_burn_tracker(workspace["burn"])[0].status == "planned"
    assert read_registry(workspace["registry"]) == []
    assert client.terminated == []


def test_a_create_failure_suggests_checking_capacity(workspace, monkeypatch, capsys):
    from ops.runpod_client import RunPodError

    _approve(workspace)
    _plan(workspace)

    class NoCapacity(FakeClient):
        def create_pod(self, **kwargs):
            raise RunPodError("no instances currently available")

    _use(monkeypatch, NoCapacity())
    gpu.main(["launch", str(workspace["spec"])])
    assert "capacity" in capsys.readouterr().err.lower()


def test_a_registry_write_failure_after_create_destroys_the_pod(
    workspace, monkeypatch, capsys
):
    # The same reasoning as an unconsumable burn row: a pod whose ceilings we could not
    # record is a pod nothing can enforce a limit on.
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())

    from ops import registry

    monkeypatch.setattr(
        gpu, "record_launch",
        lambda **kw: (_ for _ in ()).throw(registry.RegistryError("locked")))

    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert client.terminated == ["pod-new"]
    assert "could not be recorded" in capsys.readouterr().err


def test_launch_refuses_an_image_tag_that_does_not_exist(workspace, monkeypatch, capsys):
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())
    monkeypatch.setattr(gpu, "image_exists", lambda image: False)

    code = gpu.main(["launch", str(workspace["spec"])])
    assert code != 0
    assert client.created == []
    error = capsys.readouterr().err
    assert "image" in error.lower()


def test_launch_proceeds_when_the_image_cannot_be_checked(workspace, monkeypatch, capsys):
    # Docker Hub being unreachable must not block a gate.
    _approve(workspace)
    _plan(workspace)
    client = _use(monkeypatch, FakeClient())
    monkeypatch.setattr(gpu, "image_exists", lambda image: None)

    assert gpu.main(["launch", str(workspace["spec"])]) == 0
    assert len(client.created) == 1
    assert "not verified" in capsys.readouterr().err.lower()


# --- a run that self-destroyed is already closed ----------------------------------
#
# Found 6 Nov, after m1-pilot-9/10/11 all self-destroyed as designed (D-027) and all
# three left their registry rows behind. Self-destruct is the *intended* ending for a
# healthy run, so `destroy` must have a path for "pod already gone, run already closed"
# or the registry leaks a ghost row on every success. The old code had none: it tried
# `done -> aborted`, which `set_status` rightly refuses, and then reported failure while
# telling the operator to record a cost that was already recorded.


def _close(workspace, run_id="m1-pilot", cost_inr=36.52):
    """Walk a planned row to `done` with a cost, the way a finished run leaves it."""
    from ops.burn import set_status

    set_status(workspace["burn"], run_id, "running")
    set_status(workspace["burn"], run_id, "done", actual_hours=0.70, cost_inr=cost_inr)


def test_destroy_clears_the_registry_row_of_a_run_that_already_self_destroyed(
    workspace, monkeypatch
):
    _plan(workspace)
    _close(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    _use(monkeypatch, FakeClient())

    assert gpu.main(["destroy", "m1-pilot"]) == 0
    assert read_registry(workspace["registry"]) == []


def test_destroy_leaves_an_already_closed_runs_cost_and_status_alone(
    workspace, monkeypatch
):
    from ops.burn import read_burn_tracker

    _plan(workspace)
    _close(workspace, cost_inr=36.52)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    _use(monkeypatch, FakeClient())

    gpu.main(["destroy", "m1-pilot"])

    row = read_burn_tracker(workspace["burn"])[0]
    # Re-billing wall-clock from launch to now is the failure this guards: by the time
    # anyone tidies up, "now" can be many hours after a pod that lived minutes.
    assert row.status == "done"
    assert row.cost_inr == pytest.approx(36.52)


def test_destroy_says_the_run_was_already_closed_rather_than_asking_for_a_cost(
    workspace, monkeypatch, capsys
):
    _plan(workspace)
    _close(workspace)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW,
                  path=workspace["registry"])
    _use(monkeypatch, FakeClient())

    gpu.main(["destroy", "m1-pilot"])
    out = capsys.readouterr()
    assert "by hand" not in (out.out + out.err)
    assert "already" in (out.out + out.err).lower()
