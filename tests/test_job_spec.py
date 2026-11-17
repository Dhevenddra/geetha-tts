"""Tests for ops.spec — validating a job spec before anything can be launched.

A spec names a GPU, an hour ceiling and a rupee ceiling (runbook section 9). It is
the last readable description of what we are about to pay for, so a malformed one
must fail here rather than on a pod that is already billing.
"""
import pytest

from ops.spec import JobSpec, SpecError

GOOD = {
    "run_id": "m1-pilot-stage-a",
    "gate": "M1",
    "provider": "runpod",
    "gpu_type": "NVIDIA GeForce RTX 4090",
    "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1",
    "disk_gb": 100,
    "max_hours": 2.0,
    "max_cost_inr": 150.0,
    "command": "bash train_stage_a.sh",
    "data_repos": ["Dhevenddra/geetha-tts-stagea-private"],
    "checkpoint_repo": "Dhevenddra/geetha-ckpt-private",
    "wandb_project": "geetha",
}


def _raw(**overrides) -> dict:
    """A valid spec dict with fields overridden, for the candidate tests below."""
    return {**GOOD, **overrides}


def test_a_complete_spec_validates():
    spec = JobSpec.from_dict(GOOD)
    assert spec.run_id == "m1-pilot-stage-a"
    assert spec.max_cost_inr == 150.0


def test_a_missing_field_is_named_in_the_error():
    bad = {k: v for k, v in GOOD.items() if k != "max_cost_inr"}
    with pytest.raises(SpecError, match="max_cost_inr"):
        JobSpec.from_dict(bad)


def test_an_unknown_provider_is_refused():
    with pytest.raises(SpecError, match="provider"):
        JobSpec.from_dict({**GOOD, "provider": "mystery-cloud"})


def test_a_gate_outside_m1_to_m6_is_refused():
    with pytest.raises(SpecError, match="gate"):
        JobSpec.from_dict({**GOOD, "gate": "M9"})


def test_zero_or_negative_hour_ceiling_is_refused():
    with pytest.raises(SpecError, match="max_hours"):
        JobSpec.from_dict({**GOOD, "max_hours": 0})


def test_a_paid_job_with_no_rupee_ceiling_is_refused():
    with pytest.raises(SpecError, match="max_cost_inr"):
        JobSpec.from_dict({**GOOD, "max_cost_inr": 0})


def test_a_kaggle_job_must_cost_nothing():
    # Kaggle is the free tier. A kaggle spec carrying a rupee ceiling means someone
    # confused the provider, and confusing the provider is how money gets spent.
    with pytest.raises(SpecError, match="kaggle"):
        JobSpec.from_dict({**GOOD, "provider": "kaggle"})
    spec = JobSpec.from_dict({**GOOD, "provider": "kaggle", "max_cost_inr": 0})
    assert spec.max_cost_inr == 0


def test_a_run_id_that_is_not_a_safe_directory_name_is_refused():
    # run_id names a folder under E:\Geetha\runs\ and appears in shell commands.
    for bad_id in ("../escape", "m1 pilot", "m1;rm -rf /", "m1/pilot"):
        with pytest.raises(SpecError, match="run_id"):
            JobSpec.from_dict({**GOOD, "run_id": bad_id})


def test_a_spec_loads_from_yaml(tmp_path):
    import yaml

    path = tmp_path / "m1.yaml"
    path.write_text(yaml.safe_dump(GOOD), encoding="utf-8")
    assert JobSpec.from_yaml(path).gate == "M1"


def test_an_unexpected_extra_field_is_refused(tmp_path):
    # A typo'd key silently ignored is a ceiling silently not applied.
    with pytest.raises(SpecError, match="max_cost"):
        JobSpec.from_dict({**GOOD, "max_cost": 9999})


# --- cloud type belongs in the spec ---------------------------------------------
#
# Measured live on 27 Oct: an L4 create was refused on COMMUNITY and accepted on
# SECURE minutes apart. Availability differs by cloud type as well as by GPU, so which
# one a job uses is a per-job decision with a price attached — exactly the kind of thing
# a spec records rather than the client hardcoding.


def test_a_spec_defaults_to_the_cheaper_cloud_type():
    # Community is what the budget assumes, so it has to be what you get by accident.
    spec = JobSpec.from_dict(dict(GOOD))
    assert spec.cloud_type == "COMMUNITY"


def test_a_spec_can_ask_for_secure_capacity():
    spec = JobSpec.from_dict(dict(GOOD) | {"cloud_type": "SECURE"})
    assert spec.cloud_type == "SECURE"


def test_an_unknown_cloud_type_is_refused():
    with pytest.raises(SpecError, match="cloud_type"):
        JobSpec.from_dict(dict(GOOD) | {"cloud_type": "cheapest"})


def test_leaving_cloud_type_out_is_not_a_missing_field():
    # Fields with a default must not be reported as missing, or every existing spec
    # breaks the moment one is added.
    JobSpec.from_dict(dict(GOOD))


def test_an_unrecognised_field_is_still_refused():
    # The strictness that catches a typo'd ceiling must survive adding an optional one.
    with pytest.raises(SpecError, match="unrecognised"):
        JobSpec.from_dict(dict(GOOD) | {"max_cost_imr": 300})


class TestGpuCandidates:
    """D-045: the GPU is chosen at launch from live stock, not fixed in the spec.

    `gpu_type` stays required and becomes the last resort, so every spec already on
    disk keeps working and a candidates list is purely additive.
    """

    def test_a_spec_without_candidates_offers_just_its_gpu_type(self):
        spec = JobSpec.from_dict(_raw())
        assert spec.candidate_gpus == ("NVIDIA GeForce RTX 4090",)

    def test_candidates_come_first_and_the_gpu_type_is_the_last_resort(self):
        spec = JobSpec.from_dict(_raw(gpu_candidates=["NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090"]))
        assert spec.candidate_gpus == (
            "NVIDIA RTX A6000", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090")

    def test_the_gpu_type_is_not_repeated_if_it_is_already_a_candidate(self):
        spec = JobSpec.from_dict(
            _raw(gpu_candidates=["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"]))
        assert spec.candidate_gpus == ("NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000")

    def test_order_is_preserved_because_order_is_preference(self):
        names = ["NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA GeForce RTX 3090"]
        assert list(JobSpec.from_dict(_raw(gpu_candidates=names)).candidate_gpus)[:3] == names

    def test_an_empty_candidate_name_is_refused(self):
        with pytest.raises(SpecError, match="gpu_candidates"):
            JobSpec.from_dict(_raw(gpu_candidates=["NVIDIA RTX A6000", "  "]))

    def test_a_non_list_candidates_field_is_refused(self):
        with pytest.raises(SpecError, match="gpu_candidates"):
            JobSpec.from_dict(_raw(gpu_candidates="NVIDIA RTX A6000"))

    def test_the_hourly_ceiling_comes_from_the_gate_not_the_spec_author(self):
        spec = JobSpec.from_dict(_raw(max_cost_inr=300.0, max_hours=4.0))
        assert spec.ceiling_inr_per_hour == pytest.approx(75.0)
