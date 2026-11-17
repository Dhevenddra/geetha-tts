"""Tests for ops.secrets.

the project rules: the contents of .secrets\geetha.env are never printed, logged or
committed. So the loader must make leaking hard, not merely avoidable -- its repr
must not contain a value, and a missing variable must be reported by name only.
"""
import pytest

from ops.secrets import SecretStore, load_secrets

SAMPLE = """\
# Geetha keys
HF_TOKEN="hf_abcdefghijklmnop"
WANDB_API_KEY=wandb_super_secret

RUNPOD_API_KEY='rp_zzzzzzzz'
"""


def store(tmp_path) -> SecretStore:
    path = tmp_path / "geetha.env"
    path.write_text(SAMPLE, encoding="utf-8")
    return load_secrets(path)


def test_values_are_read_with_quotes_stripped(tmp_path):
    assert store(tmp_path).get("HF_TOKEN") == "hf_abcdefghijklmnop"
    assert store(tmp_path).get("RUNPOD_API_KEY") == "rp_zzzzzzzz"


def test_comments_and_blank_lines_are_ignored(tmp_path):
    assert store(tmp_path).names() == ("HF_TOKEN", "RUNPOD_API_KEY", "WANDB_API_KEY")


def test_the_repr_never_contains_a_secret_value(tmp_path):
    text = repr(store(tmp_path)) + str(store(tmp_path))
    assert "hf_abcdefghijklmnop" not in text
    assert "wandb_super_secret" not in text
    assert "rp_zzzzzzzz" not in text
    assert "HF_TOKEN" in text  # names are safe and useful


def test_a_missing_variable_raises_naming_only_the_variable(tmp_path):
    with pytest.raises(KeyError, match="VAST_API_KEY"):
        store(tmp_path).require("VAST_API_KEY")


def test_a_present_variable_is_returned_by_require(tmp_path):
    assert store(tmp_path).require("HF_TOKEN") == "hf_abcdefghijklmnop"


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_secrets(tmp_path / "absent.env")


def test_has_reports_presence_without_revealing_anything(tmp_path):
    s = store(tmp_path)
    assert s.has("HF_TOKEN") is True
    assert s.has("VAST_API_KEY") is False


def test_an_empty_value_counts_as_absent(tmp_path):
    # A key left as "HF_TOKEN=" is a half-finished setup, not a usable token.
    path = tmp_path / "e.env"
    path.write_text("HF_TOKEN=\n", encoding="utf-8")
    assert load_secrets(path).has("HF_TOKEN") is False
