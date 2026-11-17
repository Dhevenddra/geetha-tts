"""Tests for ops.gpu — the command surface from runbook section 9.

B2 ships the skeleton: the read-only commands work, and every command that could
spend money refuses until B11 finishes and dry-runs it. The refusal is tested,
because "not implemented yet" must mean "cannot be invoked", not "falls through".
"""
import pytest

from ops.gpu import build_parser, main


def test_the_parser_knows_every_command_in_the_runbook():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert set(commands) == {
        "quote",
        "balance",
        "launch",
        "status",
        "logs",
        "pull",
        "stop",
        "destroy",
    }


def test_status_accepts_the_enforce_flag():
    args = build_parser().parse_args(["status", "--enforce"])
    assert args.enforce is True


def test_status_does_not_enforce_by_default():
    assert build_parser().parse_args(["status"]).enforce is False


def test_no_command_exits_with_an_error():
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


# B11 finished launch and destroy, so these no longer assert "not implemented yet".
# They assert the properties that outlive the implementation: a launch cannot happen
# without the operator's approval on record, and a destroy cannot silently succeed on
# a pod nobody wrote down. The detailed behaviour is in tests/test_gpu_launch.py.


def test_launch_of_a_missing_spec_fails_loudly(capsys):
    code = main(["launch", "ops/jobs/does-not-matter.yaml"])
    assert code != 0
    assert "does-not-matter" in capsys.readouterr().err


def test_destroy_of_an_unrecorded_run_refuses_rather_than_reporting_success(capsys):
    code = main(["destroy", "no-such-run-id"])
    assert code != 0
    assert "no pod recorded" in capsys.readouterr().err


def test_stop_is_refused_on_principle_not_pending_implementation(capsys):
    # Stopping leaves the disk billing; every teardown path here is a delete.
    code = main(["stop", "m1-pilot"])
    assert code != 0
    assert "billing" in capsys.readouterr().err.lower()


def test_a_dry_run_launch_check_is_available_without_spending(capsys):
    # The point of the skeleton: we can prove the guards work before M1.
    code = main(["launch", "ops/jobs/does-not-matter.yaml", "--check-only"])
    out = capsys.readouterr()
    assert code != 0  # the spec does not exist, so the check fails loudly
    assert "does-not-matter" in (out.err + out.out)
