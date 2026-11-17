"""The generated wrapper must be a script bash can actually parse.

This is the bug that cost the project its first real week of GPU money, and nothing in
1,060 tests would have caught it, because every test asserted on the wrapper as a
*string* -- substrings present, ordering right -- and never once asked a shell to read
it.

`build_wrapper` interpolated the job command with `{command!r}`. Python's repr renders
an embedded single quote as \\' , which is correct Python and wrong shell: inside a
single-quoted string a backslash is literal, so bash closes the string at that quote and
everything after it is reparsed as code. The m1-pilot command quotes its repo names --
`snapshot_download('Dhevenddra/geetha-code-private', ...)` -- so the pod received

    syntax error near unexpected token `)'

and died in seconds. m1-pilot-5 died this way on 4 Nov, was recorded as an image-pull
timeout, and that reading became the premise of an entire day's plan (D-047). The pod
was never pulling. It could not parse its own command.

So: these tests run `bash -n` over the real generated script. A string assertion cannot
tell a valid script from an invalid one, and only the shell's own parser is authority.
"""

import shutil
import subprocess

import pytest

from ops.wrapper import build_wrapper

bash = shutil.which("bash")


def _bash_works() -> bool:
    """On Windows, `bash` can be the WSL launcher with no Linux installed: found, but unable to run."""
    if bash is None:
        return False
    try:
        return subprocess.run([bash, "-c", "true"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_bash = pytest.mark.skipif(not _bash_works(), reason="no working bash to parse the script")


def _parse(script: str) -> subprocess.CompletedProcess:
    """Ask bash itself whether the script is syntactically valid. -n parses, never runs."""
    return subprocess.run(
        [bash, "-n"], input=script, text=True, capture_output=True, timeout=30,
    )


def _wrapper(command: str) -> str:
    return build_wrapper(
        run_id="m1-pilot-test", command=command, max_hours=1.0,
        checkpoint_repo="Dhevenddra/geetha-ckpt-private",
    )


@needs_bash
def test_a_command_containing_single_quotes_still_parses():
    """The exact shape that broke m1-pilot-5 and m1-pilot-7."""
    command = (
        "bash -lc \"set -e; python -c \\\"from huggingface_hub import snapshot_download; "
        "snapshot_download('Dhevenddra/geetha-code-private', repo_type='model', "
        "local_dir='/workspace/code')\\\"; bash /workspace/code/jobs/m1_pilot.sh\""
    )

    result = _parse(_wrapper(command))

    assert result.returncode == 0, (
        "the generated wrapper is not valid bash:\n" + result.stderr
    )


@needs_bash
@pytest.mark.parametrize("command", [
    "echo hello",
    "echo 'single quoted'",
    'echo "double quoted"',
    "python -c \"print('nested')\"",
    "echo it's got an apostrophe",
    "echo \"mixed 'inner' quotes\"",
    "echo $HOME && echo `date`",
    "echo a; echo b | grep a || true",
])
def test_the_wrapper_survives_whatever_the_command_contains(command):
    """Quoting must be the wrapper's problem, not the spec author's.

    A spec is a config file. If writing an apostrophe in it can break a GPU run in a way
    that only shows up as a dead pod and a wrong post-mortem, the config format is a trap.
    """
    result = _parse(_wrapper(command))

    assert result.returncode == 0, (
        f"command {command!r} produced a wrapper bash cannot parse:\n{result.stderr}"
    )


@needs_bash
def test_the_real_pilot_spec_generates_a_parsable_wrapper():
    """The shipped spec, not a synthetic one -- this is what actually launches."""
    from ops.spec import JobSpec

    spec = JobSpec.from_yaml("ops/jobs/m1-pilot.yaml")
    script = build_wrapper(
        run_id=spec.run_id, command=spec.command, max_hours=spec.max_hours,
        checkpoint_repo=spec.checkpoint_repo,
    )

    result = _parse(script)

    assert result.returncode == 0, (
        "ops/jobs/m1-pilot.yaml generates a wrapper bash cannot parse -- this spec "
        "would burn a pod on a syntax error:\n" + result.stderr
    )


@needs_bash
def test_the_teardown_path_is_still_present_and_parsable():
    """The self-destroy trap is the whole safety design; it must survive quoting fixes."""
    script = _wrapper("echo 'quoted'")

    assert _parse(script).returncode == 0
    assert "trap geetha_self_destroy EXIT" in script, (
        "the EXIT trap is gone -- a pod whose command fails would run until the deadline"
    )
