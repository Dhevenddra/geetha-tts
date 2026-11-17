"""Tests for ops.registry — what a launch has to write down so a watchdog can act.

``pods_to_destroy`` needs each pod's deadline and rupee ceiling. RunPod does not know
either: they are our policy, not its. So launch records them locally, and a live pod
with no record is treated as an escapee rather than as a pod with no limits.
"""
from datetime import datetime, timedelta, timezone

import pytest

from ops.guards import pods_to_destroy
from ops.registry import (
    REGISTRY_COLUMNS,
    RegistryError,
    forget_pod,
    pod_states,
    read_registry,
    record_launch,
)
from ops.runpod_client import RunPodPod

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def _pod(pod_id="pod-1", name="m1-pilot", cost=0.34, uptime=3600):
    return RunPodPod(
        pod_id=pod_id, name=name, desired_status="RUNNING", gpu_type="NVIDIA GeForce RTX 4090",
        gpu_count=1, cost_per_hr=cost, uptime_seconds=uptime, raw={},
    )


def _registry(tmp_path):
    return tmp_path / "pod_registry.csv"


# --- recording -------------------------------------------------------------------


def test_a_launch_is_recorded_with_its_deadline_and_ceiling(tmp_path):
    path = _registry(tmp_path)
    record_launch(
        run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
        max_hours=2.0, max_cost_inr=300.0, launched_at=NOW, path=path,
    )
    rows = read_registry(path)
    assert len(rows) == 1
    assert rows[0].run_id == "m1-pilot"
    assert rows[0].pod_id == "pod-1"
    assert rows[0].max_cost_inr == 300.0
    assert rows[0].deadline == NOW + timedelta(hours=2)


def test_the_deadline_is_stored_absolute_not_as_a_duration(tmp_path):
    # A duration would restart every time something re-read the file, which is exactly
    # how a pod outlives its deadline.
    path = _registry(tmp_path)
    record_launch(run_id="r", pod_id="p", gate="M1", provider="runpod", max_hours=1.5,
                  max_cost_inr=100.0, launched_at=NOW, path=path)
    assert "2026-11-01T13:30" in path.read_text(encoding="utf-8")


def test_recording_twice_for_one_run_id_is_refused(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="r", pod_id="p1", gate="M1", provider="runpod", max_hours=1.0,
                  max_cost_inr=100.0, launched_at=NOW, path=path)
    with pytest.raises(RegistryError, match="already"):
        record_launch(run_id="r", pod_id="p2", gate="M1", provider="runpod", max_hours=1.0,
                      max_cost_inr=100.0, launched_at=NOW, path=path)


def test_a_destroyed_pod_is_forgotten_so_the_run_id_can_be_relaunched(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="r", pod_id="p1", gate="M1", provider="runpod", max_hours=1.0,
                  max_cost_inr=100.0, launched_at=NOW, path=path)
    forget_pod("p1", path=path)  # keyed on the pod id
    assert read_registry(path) == []
    record_launch(run_id="r", pod_id="p2", gate="M1", provider="runpod", max_hours=1.0,
                  max_cost_inr=100.0, launched_at=NOW, path=path)
    assert read_registry(path)[0].pod_id == "p2"


def test_forgetting_a_pod_that_was_never_recorded_is_not_an_error(tmp_path):
    # Teardown has to be idempotent: it runs on paths where the record may already be
    # gone, and raising there would strand a pod.
    forget_pod("never-existed", path=_registry(tmp_path))


def test_a_missing_registry_reads_as_empty(tmp_path):
    assert read_registry(_registry(tmp_path)) == []


def test_a_column_mismatch_is_refused(tmp_path):
    path = _registry(tmp_path)
    path.write_text("run_id,pod_id\nr,p\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="column"):
        read_registry(path)


def test_the_columns_are_the_ones_the_watchdog_reads():
    assert REGISTRY_COLUMNS == (
        "run_id", "pod_id", "gate", "provider",
        "launched_at", "deadline", "max_cost_inr", "rate_usd_per_h",
    )


# --- joining the provider's view to our policy ------------------------------------


def test_a_live_pod_gets_its_recorded_ceiling_and_deadline(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    states = pod_states([_pod(uptime=1800)], read_registry(path), inr_per_usd=88.0)
    assert len(states) == 1
    assert states[0].run_id == "m1-pilot"
    assert states[0].max_cost_inr == 300.0
    assert states[0].hours_used == pytest.approx(0.5)
    # 0.5 h at $0.34 = $0.17 = INR 14.96
    assert states[0].cost_inr == pytest.approx(0.5 * 0.34 * 88.0)


def test_a_live_pod_with_no_record_is_given_a_zero_ceiling_so_it_is_destroyed(tmp_path):
    # The dangerous pod is the one nobody wrote down. It must not come back with
    # "no limits recorded" and therefore survive.
    states = pod_states([_pod(pod_id="stray", name="whatever")], [], inr_per_usd=88.0)
    assert states[0].max_cost_inr == 0.0
    doomed = pods_to_destroy(states, burn_rows=[], now=NOW)
    assert len(doomed) == 1


def test_an_unrecorded_pods_label_mentions_its_name_so_the_reason_is_readable(tmp_path):
    # The name is kept for the log line only. It is wrapped so the label cannot be
    # mistaken for -- or matched against -- a real run_id.
    states = pod_states([_pod(pod_id="stray", name="m2-stage-a")], [], inr_per_usd=88.0)
    assert "m2-stage-a" in states[0].run_id
    assert states[0].run_id != "m2-stage-a"


def test_an_unrecorded_unnamed_pod_still_produces_a_state(tmp_path):
    states = pod_states([_pod(pod_id="stray", name="")], [], inr_per_usd=88.0)
    assert "stray" in states[0].run_id
    assert "unnamed" in states[0].run_id


def test_a_recorded_pod_past_its_deadline_is_doomed(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=1.0, max_cost_inr=300.0, launched_at=NOW - timedelta(hours=3),
                  path=path)
    from ops.guards import BurnRow

    burn = [BurnRow(run_id="m1-pilot", gate="M1", status="running",
                    planned_hours=1.0, rate_usd_per_h=0.34, cost_inr=0.0)]
    states = pod_states([_pod()], read_registry(path), inr_per_usd=88.0)
    doomed = pods_to_destroy(states, burn, now=NOW)
    assert len(doomed) == 1
    assert "deadline" in doomed[0][1]


def test_a_recorded_pod_inside_its_limits_is_left_alone(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=4.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    from ops.guards import BurnRow

    burn = [BurnRow(run_id="m1-pilot", gate="M1", status="running",
                    planned_hours=4.0, rate_usd_per_h=0.34, cost_inr=0.0)]
    states = pod_states([_pod(uptime=600)], read_registry(path), inr_per_usd=88.0)
    assert pods_to_destroy(states, burn, now=NOW) == []


# --- the registry must not fail open on a pod name -------------------------------
#
# Flagged by security review, 27 Oct. The name fallback was written so an escaped pod
# would have a readable label; its actual effect was to hand that pod the *ceiling and
# deadline* of the run whose name it matched, so a stray pod called "m1-pilot" inherited
# a INR 300 budget and a future deadline and survived the watchdog. Pod names are set at
# create time and can be reused by hand in the console, so this is settable by accident.


def test_a_pod_whose_id_is_unrecorded_does_not_inherit_limits_from_a_name_match(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-real", gate="M1", provider="runpod",
                  max_hours=4.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    # Same name, different pod: not the pod we launched.
    stray = _pod(pod_id="pod-stray", name="m1-pilot")
    states = pod_states([stray], read_registry(path), inr_per_usd=88.0)
    assert states[0].max_cost_inr == 0.0
    assert states[0].deadline < NOW


def test_such_a_pod_is_destroyed_even_though_its_name_is_in_the_burn_tracker(tmp_path):
    from ops.guards import BurnRow

    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-real", gate="M1", provider="runpod",
                  max_hours=4.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    burn = [BurnRow(run_id="m1-pilot", gate="M1", status="running",
                    planned_hours=4.0, rate_usd_per_h=0.34, cost_inr=0.0)]
    states = pod_states([_pod(pod_id="pod-stray", name="m1-pilot")],
                        read_registry(path), inr_per_usd=88.0)
    assert len(pods_to_destroy(states, burn, now=NOW)) == 1


def test_an_unrecorded_pods_label_can_never_match_a_burn_tracker_run_id(tmp_path):
    # The label is for humans. It must not double as an identity that could let a pod
    # pass the "is this run authorised?" check.
    states = pod_states([_pod(pod_id="stray", name="m1-pilot")], [], inr_per_usd=88.0)
    assert states[0].run_id != "m1-pilot"
    assert "stray" in states[0].run_id
    assert "m1-pilot" in states[0].run_id  # still readable


def test_the_recorded_pod_itself_still_gets_its_limits(tmp_path):
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-real", gate="M1", provider="runpod",
                  max_hours=4.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    states = pod_states([_pod(pod_id="pod-real", name="m1-pilot")],
                        read_registry(path), inr_per_usd=88.0)
    assert states[0].run_id == "m1-pilot"
    assert states[0].max_cost_inr == 300.0


def test_forgetting_is_keyed_on_the_pod_id_not_the_run_id(tmp_path):
    # Two rows sharing a run_id must not both vanish when one pod is destroyed: the
    # survivor would lose its recorded ceiling and become invisible to the watchdog.
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-a", gate="M1", provider="runpod",
                  max_hours=1.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    forget_pod("pod-a", path=path)
    assert read_registry(path) == []


# --- the rate actually paid is recorded, not the one that was planned ------------
#
# Second security review, 27 Oct. destroy priced a run from the burn row's *planned*
# rate, so a price that moved up within the 20% tolerance recorded up to 20% less than
# the run cost — and a row with no rate recorded the run as free, which hands its share
# of the gate's authorisation back. The rate we launched at is known at launch, so it
# is written down there.


def test_the_quoted_rate_is_recorded_at_launch(tmp_path):
    path = _registry(tmp_path)
    row = record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                        max_hours=2.0, max_cost_inr=300.0, rate_usd_per_h=0.41,
                        launched_at=NOW, path=path)
    assert row.rate_usd_per_h == 0.41
    assert read_registry(path)[0].rate_usd_per_h == 0.41


def test_the_rate_column_is_part_of_the_schema():
    assert "rate_usd_per_h" in REGISTRY_COLUMNS


def test_a_row_without_a_rate_reads_as_zero_rather_than_failing(tmp_path):
    # Forward compatibility only; the caller is expected to notice a zero rate, since
    # pricing a run at zero is how spend goes unrecorded.
    path = _registry(tmp_path)
    record_launch(run_id="m1-pilot", pod_id="pod-1", gate="M1", provider="runpod",
                  max_hours=2.0, max_cost_inr=300.0, launched_at=NOW, path=path)
    assert read_registry(path)[0].rate_usd_per_h == 0.0
