"""Tests for geetha.runlog — provenance for every ETL and eval output.

Runbook rule 9: deterministic ETL, fixed seeds, and every output records the hash
of its input manifest. Rule in the project rules: secrets are never printed or logged, so
the logger has to actively refuse them -- a provenance file is exactly the kind of
artefact that gets pasted into an issue.
"""
import hashlib
import json

import pytest

from geetha.runlog import RunLog, file_sha256, text_hash


def test_texts_that_normalise_alike_hash_alike():
    # This is the leak hash of runbook rule 3: it must see through punctuation.
    assert text_hash("வணக்கம், நண்பா!") == text_hash("வணக்கம் நண்பா")


def test_different_sentences_hash_differently():
    assert text_hash("ஒன்று") != text_hash("இரண்டு")


def test_text_hash_is_stable_across_processes():
    # Hard-coded so a change to norm_text that silently alters every manifest hash
    # fails here instead of quietly invalidating the frozen eval vault.
    assert text_hash("தமிழ்") == hashlib.sha256("தமிழ்".encode()).hexdigest()


def test_file_sha256_matches_the_bytes_on_disk(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"utt_id,text\n1,hello\n")
    assert file_sha256(path) == hashlib.sha256(b"utt_id,text\n1,hello\n").hexdigest()


def test_a_run_records_the_hash_of_every_input(tmp_path):
    manifest = tmp_path / "in.csv"
    manifest.write_bytes(b"a,b\n")
    log = RunLog(run_id="etl-step2", seed=1234, out_dir=tmp_path)
    log.add_input(manifest)
    assert log.inputs[str(manifest)] == file_sha256(manifest)


def test_a_finished_run_writes_json_that_reads_back(tmp_path):
    log = RunLog(run_id="etl-step2", seed=1234, out_dir=tmp_path)
    log.add_param("stage", "A")
    written = log.finish(status="ok")
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "etl-step2"
    assert loaded["seed"] == 1234
    assert loaded["status"] == "ok"
    assert loaded["params"]["stage"] == "A"


def test_a_run_records_the_code_commit_it_ran_from(tmp_path):
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / ".git").exists():
        pytest.skip("not a git checkout (for example a downloaded zip), so there is no commit to record")
    log = RunLog(run_id="r", seed=1, out_dir=tmp_path)
    assert log.git_commit is not None and len(log.git_commit) >= 7


def test_a_parameter_that_looks_like_a_secret_is_refused(tmp_path):
    log = RunLog(run_id="r", seed=1, out_dir=tmp_path)
    for name in ("hf_token", "WANDB_API_KEY", "runpod_secret", "password"):
        with pytest.raises(ValueError, match="secret"):
            log.add_param(name, "hunter2")


def test_a_safe_parameter_with_key_in_a_longer_word_is_still_allowed(tmp_path):
    log = RunLog(run_id="r", seed=1, out_dir=tmp_path)
    log.add_param("keep_top_fraction", 0.35)
    assert log.params["keep_top_fraction"] == 0.35


def test_a_failed_run_is_recorded_as_failed(tmp_path):
    log = RunLog(run_id="r", seed=1, out_dir=tmp_path)
    written = log.finish(status="failed", note="DNSMOS filter kept only 12 h")
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["status"] == "failed"
    assert "12 h" in loaded["note"]
