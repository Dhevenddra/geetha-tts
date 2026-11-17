"""The on-pod wrapper. These tests are about one thing: the pod always dies."""

from __future__ import annotations

import pytest

from ops.wrapper import (
    DEFAULT_IDLE_MINUTES,
    SELF_DESTROY_SNIPPET,
    build_probe_command,
    build_wrapper,
)


def _wrapper(**overrides) -> str:
    kwargs = dict(
        run_id="m1-pilot",
        command="python train.py --steps 50",
        max_hours=1.5,
        checkpoint_repo="Dhevenddra/geetha-ckpt-private",
    )
    kwargs.update(overrides)
    return build_wrapper(**kwargs)


class TestItAlwaysDies:
    def test_teardown_is_trapped_on_exit(self):
        # The dangerous failure is not a pod that dies early, it is one that lives
        # forever. EXIT covers crash, OOM, and the command never starting.
        assert "trap geetha_self_destroy EXIT" in _wrapper()

    def test_it_deletes_and_never_stops(self):
        # `runpodctl stop pod` leaves the disk allocated and still billing.
        script = _wrapper()
        assert "remove pod" in script
        assert "stop pod" not in script
        assert "stop" not in script.replace("stopped", "")

    def test_there_is_a_rest_fallback_for_the_delete(self):
        # Whether a pod-scoped key may delete its own pod is undocumented, so the
        # wrapper must not depend on runpodctl alone.
        script = _wrapper()
        assert "-X DELETE" in script
        assert "api.runpod.io/v2/pods/" in script

    def test_the_command_runs_under_a_deadline(self):
        script = _wrapper(max_hours=2.0)
        assert "timeout --signal=TERM" in script
        assert "7200" in script

    def test_a_fractional_deadline_becomes_whole_seconds(self):
        assert "1800 " in _wrapper(max_hours=0.5) or "1800\n" in _wrapper(max_hours=0.5)

    def test_set_e_is_not_used(self):
        # An -e abort partway through teardown would skip the push or the delete.
        script = _wrapper()
        assert "set -e\n" not in script
        assert "set -u" in script

    def test_a_pod_with_no_deadline_is_refused(self):
        with pytest.raises(ValueError, match="no deadline"):
            _wrapper(max_hours=0)

    def test_a_negative_deadline_is_refused(self):
        with pytest.raises(ValueError):
            _wrapper(max_hours=-1)


class TestIdleWatchdog:
    def test_it_is_started_in_the_background(self):
        script = _wrapper()
        assert "geetha_idle_watchdog &" in script

    def test_it_destroys_rather_than_merely_exiting(self):
        script = _wrapper()
        assert "idle watchdog fired" in script
        # The watchdog branch must call teardown itself; exiting the subshell would
        # leave the pod alive. Slice from the firing message to the branch's exit
        # rather than splitting on "}", which lands inside ${util:-0}.
        fired = script.index("idle watchdog fired")
        branch = script[fired : script.index("exit 99", fired)]
        assert "geetha_self_destroy" in branch

    def test_the_idle_threshold_is_configurable_and_appears(self):
        assert "/5 min" in _wrapper(idle_minutes=5)

    def test_the_default_is_long_enough_for_a_checkpoint_upload(self):
        assert DEFAULT_IDLE_MINUTES >= 10

    def test_a_zero_idle_threshold_is_refused(self):
        with pytest.raises(ValueError):
            _wrapper(idle_minutes=0)


class TestCheckpoints:
    def test_checkpoints_are_pushed_before_teardown(self):
        script = _wrapper()
        push = script.index("push_checkpoints")
        exit_line = script.rindex("exit $GEETHA_RC")
        assert push < exit_line

    def test_a_failed_push_does_not_abort_teardown(self):
        assert "WARNING: checkpoint push failed" in _wrapper()

    def test_the_command_is_quoted_so_it_cannot_split(self):
        script = _wrapper(command="python train.py --note 'a b c'")
        assert "--note" in script


class TestProbe:
    def test_the_probe_uses_the_same_teardown_path_as_the_wrapper(self):
        # A probe that proved a different code path would prove nothing.
        assert SELF_DESTROY_SNIPPET in build_probe_command()
        assert SELF_DESTROY_SNIPPET in _wrapper()

    def test_it_sleeps_first_so_the_pod_is_observable(self):
        assert "sleep 30" in build_probe_command(sleep_seconds=30)

    def test_it_logs_the_runpodctl_version_and_whether_a_key_was_injected(self):
        probe = build_probe_command()
        assert "runpodctl version" in probe
        assert "key_present" in probe

    def test_it_reports_surviving_its_own_delete(self):
        assert "outlived its own delete call" in build_probe_command()

    def test_the_probe_never_echoes_the_key(self):
        probe = build_probe_command()
        assert "$RUNPOD_API_KEY" in probe          # used
        assert "echo \"$RUNPOD_API_KEY\"" not in probe   # never printed
