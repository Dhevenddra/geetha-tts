"""Tests for ops.burn — reading manifests/burn_tracker.csv.

Every paid run has a row here before it launches and an actual cost after it ends
(runbook section 9). The reader is strict: a tracker we cannot parse must stop a
launch, not be treated as "no rows found".
"""
import pytest

from ops.burn import BURN_COLUMNS, read_burn_tracker
from ops.guards import BurnRow

HEADER = ",".join(BURN_COLUMNS)


def write(tmp_path, *lines):
    path = tmp_path / "burn_tracker.csv"
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return path


def test_a_header_only_tracker_has_no_rows(tmp_path):
    assert read_burn_tracker(write(tmp_path)) == []


def test_a_planned_row_is_read(tmp_path):
    path = write(
        tmp_path,
        "2026-10-30,M1,m1-pilot,runpod,RTX 4090,4.0,,0.34,88.0,300,,0,planned,Stage A pilot,",
    )
    assert read_burn_tracker(path) == [
        BurnRow(
            run_id="m1-pilot",
            gate="M1",
            status="planned",
            planned_hours=4.0,
            rate_usd_per_h=0.34,
            cost_inr=0.0,
            max_cost_inr=300.0,
        )
    ]


def test_a_finished_row_carries_its_actual_cost(tmp_path):
    path = write(
        tmp_path,
        "2026-10-30,M1,m1-pilot,runpod,RTX 4090,4.0,3.8,0.34,88.0,300,113.70,113.70,done,Stage A pilot,runs/m1-pilot",
    )
    assert read_burn_tracker(path)[0].cost_inr == pytest.approx(113.70)


def test_an_unknown_status_is_refused(tmp_path):
    path = write(
        tmp_path,
        "2026-10-30,M1,m1-pilot,runpod,RTX 4090,4.0,,0.34,88.0,300,,0,maybe,Stage A pilot,",
    )
    with pytest.raises(ValueError, match="status"):
        read_burn_tracker(path)


def test_a_tracker_with_the_wrong_columns_is_refused(tmp_path):
    path = tmp_path / "burn_tracker.csv"
    path.write_text("gate,run_id\nM1,m1-pilot\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column"):
        read_burn_tracker(path)


def test_a_missing_tracker_is_refused_rather_than_read_as_empty(tmp_path):
    # "No file" must never look like "no planned runs, go ahead".
    with pytest.raises(FileNotFoundError):
        read_burn_tracker(tmp_path / "nope.csv")


def test_duplicate_run_ids_are_refused(tmp_path):
    row = "2026-10-30,M1,m1-pilot,runpod,RTX 4090,4.0,,0.34,88.0,300,,0,planned,Stage A pilot,"
    with pytest.raises(ValueError, match="duplicate"):
        read_burn_tracker(write(tmp_path, row, row))


# --- status transitions: one planned row authorises one launch -------------------
#
# Flagged by security review, 27 Oct. Nothing ever wrote to the burn tracker, so a row
# stayed on `planned` forever: destroy forgot the registry row and the same run could be
# relaunched indefinitely on one authorisation. It also meant the runbook's "planned,
# then actual" was only half implemented -- no actual cost was ever recorded.

import csv  # noqa: E402

import pytest  # noqa: E402

from ops.burn import BURN_COLUMNS, BurnTrackerError, set_status  # noqa: E402


def _tracker(tmp_path, *rows):
    path = tmp_path / "burn_tracker.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BURN_COLUMNS)
        for row in rows:
            writer.writerow(row)
    return path


def _planned(run_id="m1-pilot", gate="M1"):
    return ["2026-11-01", gate, run_id, "runpod", "RTX 4090", "2", "", "0.34", "88",
            "300", "", "", "planned", "pilot", ""]


def test_a_planned_row_can_be_marked_running(tmp_path):
    path = _tracker(tmp_path, _planned())
    set_status(path, "m1-pilot", "running")
    assert read_burn_tracker(path)[0].status == "running"


def test_marking_running_twice_is_refused_so_one_row_authorises_one_launch(tmp_path):
    path = _tracker(tmp_path, _planned())
    set_status(path, "m1-pilot", "running")
    with pytest.raises(BurnTrackerError, match="running"):
        set_status(path, "m1-pilot", "running")


def test_a_finished_run_records_its_actual_hours_and_cost(tmp_path):
    path = _tracker(tmp_path, _planned())
    set_status(path, "m1-pilot", "running")
    set_status(path, "m1-pilot", "done", actual_hours=1.75, cost_inr=52.4)
    row = read_burn_tracker(path)[0]
    assert row.status == "done"
    assert row.cost_inr == pytest.approx(52.4)
    assert "1.75" in path.read_text(encoding="utf-8")


def test_other_rows_are_left_exactly_as_they_were(tmp_path):
    path = _tracker(tmp_path, _planned("m1-pilot"), _planned("m2-stage-a", gate="M2"))
    set_status(path, "m1-pilot", "running")
    rows = {row.run_id: row for row in read_burn_tracker(path)}
    assert rows["m2-stage-a"].status == "planned"
    assert rows["m2-stage-a"].gate == "M2"


def test_an_unknown_run_id_is_refused_rather_than_silently_doing_nothing(tmp_path):
    # A silent no-op here would let a launch believe its row was consumed.
    path = _tracker(tmp_path, _planned())
    with pytest.raises(BurnTrackerError, match="not in the burn tracker"):
        set_status(path, "never-planned", "running")


def test_an_unknown_status_is_refused(tmp_path):
    path = _tracker(tmp_path, _planned())
    with pytest.raises(BurnTrackerError, match="status"):
        set_status(path, "m1-pilot", "finished-ish")


def test_a_run_can_be_aborted_from_planned_without_having_run(tmp_path):
    path = _tracker(tmp_path, _planned())
    set_status(path, "m1-pilot", "aborted", cost_inr=0.0)
    assert read_burn_tracker(path)[0].status == "aborted"


def test_a_done_row_cannot_be_reopened_to_planned(tmp_path):
    # Reopening is how one authorisation becomes several.
    path = _tracker(tmp_path, _planned())
    set_status(path, "m1-pilot", "running")
    set_status(path, "m1-pilot", "done", actual_hours=1.0, cost_inr=30.0)
    with pytest.raises(BurnTrackerError, match="done"):
        set_status(path, "m1-pilot", "planned")


def test_the_cumulative_column_is_kept_in_step(tmp_path):
    path = _tracker(tmp_path, _planned("a"), _planned("b"))
    set_status(path, "a", "running")
    set_status(path, "a", "done", actual_hours=1.0, cost_inr=100.0)
    set_status(path, "b", "running")
    set_status(path, "b", "done", actual_hours=1.0, cost_inr=50.0)
    text = path.read_text(encoding="utf-8").splitlines()
    assert "150" in text[2]  # b's cumulative is 100 + 50


def test_a_write_that_fails_part_way_leaves_the_tracker_intact(tmp_path, monkeypatch):
    # The tracker is the record of every rupee. Rewritten in place, a crash mid-write
    # truncates it; every later launch then refuses (fail-closed, good) but the record
    # of what was already spent is gone (not good). So it is written to a temporary
    # file and moved into place.
    import os

    path = _tracker(tmp_path, _planned())
    before = path.read_text(encoding="utf-8")

    real_replace = os.replace

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        set_status(path, "m1-pilot", "running")
    monkeypatch.setattr(os, "replace", real_replace)

    assert path.read_text(encoding="utf-8") == before
    assert read_burn_tracker(path)[0].status == "planned"
