"""Tests for ops.watchdog — the safety layer that is not on the pod.

The on-pod layers (``ops/wrapper.py``) cannot save us from a pod that has wedged badly
enough to stop running its own wrapper. This one runs on the PC, so its own failure
modes matter: it must keep going after an API error, and it must leave a trail that
``/session-start`` can check, because a watchdog nobody can prove ran is not a safety
layer.
"""
from datetime import datetime, timedelta, timezone

import pytest

from ops.runpod_client import RunPodError, RunPodPod
from ops.watchdog import (
    MAX_HEARTBEAT_AGE_MINUTES,
    WatchResult,
    heartbeat_line,
    last_heartbeat,
    watch_once,
)

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def _pod(pod_id="pod-1", name="m1-pilot", uptime=600, cost=0.34):
    return RunPodPod(pod_id, name, "RUNNING", "RTX 4090", 1, cost, uptime, {})


class FakeClient:
    def __init__(self, pods=(), fail=None):
        self._pods = list(pods)
        self._fail = fail
        self.terminated = []

    def live_pods(self):
        if self._fail:
            raise self._fail
        return list(self._pods)

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)
        return True


# --- the heartbeat ----------------------------------------------------------------


def test_a_heartbeat_is_written_even_when_nothing_is_running():
    # "No pods" and "the watchdog is dead" must never look the same in the log.
    line = heartbeat_line(NOW, WatchResult(live=0, doomed=0, destroyed=(), error=None))
    assert line.startswith("2026-11-01T12:00")
    assert "live=0" in line


def test_a_heartbeat_names_the_pods_it_destroyed():
    line = heartbeat_line(
        NOW, WatchResult(live=1, doomed=1, destroyed=("pod-1",), error=None)
    )
    assert "destroyed=pod-1" in line


def test_a_heartbeat_records_an_api_error_rather_than_omitting_the_line():
    # A silent gap in the log is indistinguishable from the machine being asleep.
    line = heartbeat_line(NOW, WatchResult(live=0, doomed=0, destroyed=(), error="HTTP 503"))
    assert "ERROR" in line
    assert "HTTP 503" in line


def test_the_last_heartbeat_is_read_back_from_the_log(tmp_path):
    log = tmp_path / "gpu_watch.log"
    log.write_text(
        heartbeat_line(NOW - timedelta(minutes=30),
                       WatchResult(live=0, doomed=0, destroyed=(), error=None)) + "\n"
        + heartbeat_line(NOW - timedelta(minutes=2),
                         WatchResult(live=1, doomed=0, destroyed=(), error=None)) + "\n",
        encoding="utf-8",
    )
    assert last_heartbeat(log) == NOW - timedelta(minutes=2)


def test_a_missing_log_has_no_last_heartbeat(tmp_path):
    assert last_heartbeat(tmp_path / "absent.log") is None


def test_an_unparseable_last_line_falls_back_to_an_earlier_one(tmp_path):
    # A line truncated by a power cut must not make the whole log unreadable.
    log = tmp_path / "gpu_watch.log"
    good = heartbeat_line(NOW - timedelta(minutes=5),
                          WatchResult(live=0, doomed=0, destroyed=(), error=None))
    log.write_text(good + "\n2026-11-01T12:0", encoding="utf-8")
    assert last_heartbeat(log) == NOW - timedelta(minutes=5)


def test_the_freshness_window_is_the_one_session_start_checks():
    # /session-start asks whether the watchdog ran in the last 15 minutes.
    assert MAX_HEARTBEAT_AGE_MINUTES == 15


# --- one pass of the loop ---------------------------------------------------------


def test_a_pod_inside_its_limits_survives_a_pass(tmp_path):
    from ops.registry import record_launch

    registry = tmp_path / "pod_registry.csv"
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=4.0, max_cost_inr=300.0, launched_at=NOW, path=registry)
    burn = _burn(tmp_path)
    client = FakeClient(pods=[_pod()])
    result = watch_once(client, now=NOW, registry_path=registry, burn_path=burn)
    assert result.live == 1
    assert result.doomed == 0
    assert client.terminated == []


def test_a_pod_past_its_deadline_is_destroyed_in_a_pass(tmp_path):
    from ops.registry import read_registry, record_launch

    registry = tmp_path / "pod_registry.csv"
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=1.0, max_cost_inr=300.0, launched_at=NOW - timedelta(hours=3),
                  path=registry)
    client = FakeClient(pods=[_pod(uptime=10800)])
    result = watch_once(client, now=NOW, registry_path=registry, burn_path=_burn(tmp_path))
    assert result.destroyed == ("pod-1",)
    assert client.terminated == ["pod-1"]
    assert read_registry(registry) == []


def test_a_pod_nobody_wrote_down_is_destroyed_in_a_pass(tmp_path):
    client = FakeClient(pods=[_pod(pod_id="stray", name="mystery", cost=1.19)])
    result = watch_once(client, now=NOW, registry_path=tmp_path / "none.csv",
                        burn_path=_burn(tmp_path))
    assert result.destroyed == ("stray",)


def test_an_api_failure_is_reported_and_does_not_raise(tmp_path):
    # The loop must survive a blip. Raising here would stop the watchdog and leave
    # whatever is running unwatched, which is worse than a missed pass.
    client = FakeClient(fail=RunPodError("HTTP 503"))
    result = watch_once(client, now=NOW, registry_path=tmp_path / "none.csv",
                        burn_path=_burn(tmp_path))
    assert result.error is not None
    assert "503" in result.error
    assert result.destroyed == ()


def test_a_failed_teardown_is_surfaced_not_swallowed(tmp_path):
    from ops.registry import record_launch

    registry = tmp_path / "pod_registry.csv"
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=1.0, max_cost_inr=300.0, launched_at=NOW - timedelta(hours=3),
                  path=registry)

    class Stubborn(FakeClient):
        def terminate_pod(self, pod_id):
            raise RunPodError("HTTP 500")

    result = watch_once(Stubborn(pods=[_pod(uptime=10800)]), now=NOW,
                        registry_path=registry, burn_path=_burn(tmp_path))
    assert result.destroyed == ()
    assert result.error is not None
    assert "pod-1" in result.error


def test_an_unreadable_burn_tracker_does_not_trigger_a_mass_teardown(tmp_path):
    # A parse error means we do not know what is authorised. Destroying everything on
    # that basis would turn a typo into a lost training run.
    burn = tmp_path / "burn_tracker.csv"
    burn.write_text("not,the,right,columns\n", encoding="utf-8")
    client = FakeClient(pods=[_pod()])
    result = watch_once(client, now=NOW, registry_path=tmp_path / "none.csv", burn_path=burn)
    assert client.terminated == []
    assert result.error is not None


def _burn(tmp_path, run_id="m1-pilot"):
    """A burn tracker carrying a row for ``run_id``.

    A pod whose run_id is absent from the tracker is destroyed on sight, which is the
    point of that rule — so a test about staying inside limits has to supply the row,
    or it is really testing the unknown-pod rule by accident.
    """
    from ops.burn import BURN_COLUMNS

    path = tmp_path / "burn_tracker.csv"
    if not path.exists():
        # Built from the column names rather than as a positional string, so adding a
        # column cannot silently shift every field one place to the left.
        row = {name: "" for name in BURN_COLUMNS}
        row.update(date="2026-11-01", gate="M1", run_id=run_id, provider="runpod",
                   gpu="RTX 4090", planned_hours="1", rate_usd_per_h="0.34",
                   inr_per_usd="88", max_cost_inr="300", status="planned", purpose="pilot")
        path.write_text(
            ",".join(BURN_COLUMNS) + "\n"
            + ",".join(row[name] for name in BURN_COLUMNS) + "\n",
            encoding="utf-8",
        )
    return path
