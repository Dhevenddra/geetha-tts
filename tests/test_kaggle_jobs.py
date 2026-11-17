"""The Kaggle job register — the thing that stops a finished run being lost."""

from __future__ import annotations

from ops.kaggle_jobs import KaggleJob, mark, pending_jobs, read_register, register_job


def _path(tmp_path):
    return tmp_path / "kaggle_jobs.csv"


class TestRegister:
    def test_a_missing_register_reads_as_empty(self, tmp_path):
        assert read_register(_path(tmp_path)) == []

    def test_a_registered_job_round_trips(self, tmp_path):
        path = _path(tmp_path)
        register_job("me/c1", "contamination check", r"E:\runs\c1", path=path)
        jobs = read_register(path)
        assert len(jobs) == 1
        assert jobs[0].slug == "me/c1"
        assert jobs[0].result_path == r"E:\runs\c1"
        assert jobs[0].status == "pushed"

    def test_a_new_job_starts_open(self, tmp_path):
        path = _path(tmp_path)
        register_job("me/c1", "x", "y", path=path)
        assert [job.slug for job in pending_jobs(path)] == ["me/c1"]

    def test_re_pushing_a_slug_replaces_its_row(self, tmp_path):
        # Kaggle keeps one kernel per slug, so two open rows could only disagree.
        path = _path(tmp_path)
        register_job("me/c1", "first", "p1", path=path, now="2026-10-27 09:00 UTC")
        register_job("me/c1", "second", "p2", path=path, now="2026-10-27 11:00 UTC")
        jobs = read_register(path)
        assert len(jobs) == 1
        assert jobs[0].purpose == "second"
        assert jobs[0].pushed_at == "2026-10-27 11:00 UTC"

    def test_collecting_closes_the_job_and_stamps_the_time(self, tmp_path):
        path = _path(tmp_path)
        register_job("me/c1", "x", "y", path=path)
        mark("me/c1", "collected", path=path)
        assert pending_jobs(path) == []
        assert read_register(path)[0].collected_at

    def test_a_failed_job_is_also_closed(self, tmp_path):
        path = _path(tmp_path)
        register_job("me/c1", "x", "y", path=path)
        mark("me/c1", "failed", notes="OOM on T4", path=path)
        assert pending_jobs(path) == []
        assert read_register(path)[0].notes == "OOM on T4"

    def test_other_jobs_are_untouched_by_a_mark(self, tmp_path):
        path = _path(tmp_path)
        register_job("me/a", "a", "pa", path=path)
        register_job("me/b", "b", "pb", path=path)
        mark("me/a", "collected", path=path)
        assert [job.slug for job in pending_jobs(path)] == ["me/b"]

    def test_unicode_in_a_purpose_survives(self, tmp_path):
        path = _path(tmp_path)
        register_job("me/c1", "தமிழ் check", "p", path=path)
        assert read_register(path)[0].purpose == "தமிழ் check"


class TestOpenness:
    def test_running_counts_as_open(self):
        job = KaggleJob("t", "s", "p", "running", "T4x2", "r", "", "")
        assert job.is_open

    def test_collected_does_not(self):
        job = KaggleJob("t", "s", "p", "collected", "T4x2", "r", "t", "")
        assert not job.is_open
