"""The operator's run monitor: what it says must be true, because he acts on it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "watch_run", Path(__file__).resolve().parents[1] / "scripts" / "watch_run.py")
watch_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watch_run)

verdict, stage_view, crumbs_for = watch_run.verdict, watch_run.stage_view, watch_run.crumbs_for


def kind(**kw):
    defaults = dict(pod_alive=True, ever_alive=True, crumbs=[], stage_a="absent",
                    stage_b="absent")
    return verdict(**{**defaults, **kw})[0]


class TestVerdict:
    def test_done_needs_both_the_final_adapter_and_the_pod_gone(self):
        assert kind(stage_a="final", stage_b="final", pod_alive=False) == "done"
        assert kind(stage_a="final", stage_b="final", pod_alive=True) == "running"

    def test_a_vanished_pod_with_unfinished_training_is_a_problem(self):
        assert kind(pod_alive=False, stage_a="running") == "problem"
        assert kind(pod_alive=False, stage_a="final", stage_b="running") == "problem"

    def test_a_pod_never_seen_and_no_breadcrumbs_is_not_yet_a_problem(self):
        assert kind(pod_alive=False, ever_alive=False) == "running"

    def test_a_pod_never_seen_but_with_breadcrumbs_did_exist_and_is_gone(self):
        assert kind(pod_alive=False, ever_alive=False,
                    crumbs=["01-container-started"]) == "problem"

    def test_a_failed_rehearsal_is_a_problem_even_while_the_pod_lingers(self):
        assert kind(crumbs=["04-rehearsal-FAILED-1"]) == "problem"

    def test_a_stage_that_exited_non_zero_is_a_problem(self):
        assert kind(crumbs=["05-stage-a-exit-1"]) == "problem"
        assert kind(crumbs=["05-stage-a-exit-0"], stage_a="final") == "running"

    def test_the_message_names_the_stage_that_is_training(self):
        assert "Stage A" in verdict(pod_alive=True, ever_alive=True,
                                    crumbs=["04-starting-stage-a"], stage_a="running",
                                    stage_b="absent")[1]
        assert "Stage B" in verdict(pod_alive=True, ever_alive=True, crumbs=[],
                                    stage_a="final", stage_b="running")[1]


class TestStageView:
    FILES = ["m2-main/stageA/ckpt-0000120/trainer_state.json",
             "m2-main/stageA/ckpt-0000480/trainer_state.json",
             "m2-main/stageA/ckpt-0000480/adapter_model.safetensors",
             "rehearsal/m2-main-1/stageA/final/trainer_state.json"]

    def test_the_newest_checkpoint_is_reported(self):
        assert stage_view(self.FILES, "m2-main", "A") == (
            "running", "m2-main/stageA/ckpt-0000480/trainer_state.json")

    def test_the_rehearsal_is_never_mistaken_for_the_real_run(self):
        assert stage_view(self.FILES, "m2-main", "B") == ("absent", None)
        assert stage_view(self.FILES, "m2-main", "A")[0] != "final"

    def test_final_wins(self):
        files = self.FILES + ["m2-main/stageA/final/trainer_state.json"]
        assert stage_view(files, "m2-main", "A")[0] == "final"


def test_breadcrumbs_are_read_for_this_run_only():
    files = ["breadcrumbs/m2-main-1/01-container-started.txt",
             "breadcrumbs/m1-pilot-11/04-starting-stage-a.txt"]
    assert crumbs_for(files, "m2-main-1") == ["01-container-started"]
