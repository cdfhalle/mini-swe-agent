"""Tests for the enroot environment.

The bulk of these mock ``subprocess.run`` so the argv construction is verified
without enroot (or an image) being present -- the interesting logic in this
environment is entirely in how the ``enroot start`` command line is assembled.
Tests that need a real runtime are marked ``slow`` and skipped when enroot is
missing.
"""

import os
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from minisweagent.environments.enroot import EnrootEnvironment, EnrootEnvironmentConfig
from minisweagent.exceptions import Submitted


def is_enroot_available() -> bool:
    try:
        subprocess.run(["enroot", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def run_mock():
    """Patch subprocess.run inside the enroot module with a success result."""
    with patch("minisweagent.environments.enroot.subprocess.run") as m:
        m.return_value = MagicMock(stdout="", returncode=0)
        yield m


def _borrowed(**kwargs) -> EnrootEnvironment:
    """An environment that borrows an existing container (skips `enroot create`).

    `explicit_shell` is pinned so these tests never depend on a rootfs existing
    on disk; the detection itself is covered separately below.
    """
    kwargs.setdefault("explicit_shell", False)
    return EnrootEnvironment(image="/img.sqsh", container_name="ctr", **kwargs)


def _start_argv(run_mock) -> list[str]:
    """argv of the last `enroot start` call."""
    for c in reversed(run_mock.call_args_list):
        argv = c.args[0]
        if "start" in argv:
            return argv
    raise AssertionError(f"no `enroot start` call in {run_mock.call_args_list}")


# --------------------------------------------------------------------------- config


def test_config_defaults():
    config = EnrootEnvironmentConfig(image="/scratch/img.sqsh")

    assert config.image == "/scratch/img.sqsh"
    assert config.container_name == ""
    assert config.cwd == "/"
    assert config.env == {}
    assert config.forward_env == []
    assert config.mounts == []
    assert config.timeout == 30
    assert config.executable == "enroot"
    assert config.create_retries == 3
    # --rw is required: without it, file edits do not persist between commands.
    assert config.start_args == ["--rw"]
    assert config.shell == "bash"
    assert config.apply_image_env is True


# --------------------------------------------------------------------- container life


def test_creates_container_when_no_name_given(run_mock):
    env = EnrootEnvironment(image="/img.sqsh", explicit_shell=False)

    argv = run_mock.call_args_list[0].args[0]
    assert argv[:3] == ["enroot", "create", "--name"]
    assert argv[3] == env.container_name
    assert argv[4] == "/img.sqsh"
    assert env.container_name.startswith("minisweagent-")
    assert env._owns_container is True


def test_borrowed_container_is_not_created(run_mock):
    env = _borrowed()

    assert env.container_name == "ctr"
    assert env._owns_container is False
    assert run_mock.call_args_list == []


def test_borrowed_container_is_not_removed_on_cleanup(run_mock):
    _borrowed().cleanup()

    assert run_mock.call_args_list == []


def test_owned_container_is_removed_on_cleanup(run_mock):
    env = EnrootEnvironment(image="/img.sqsh", explicit_shell=False)
    run_mock.reset_mock()

    env.cleanup()

    assert run_mock.call_args_list == [
        call(["enroot", "remove", "-f", env.container_name], capture_output=True)
    ]


def _failing_create(n_failures: int):
    """subprocess.run stand-in where the first `n_failures` creates fail.

    Keyed on the enroot subcommand rather than call order, so `remove` (including
    the one `__del__` issues at GC time) always succeeds and never exhausts.
    """
    state = {"creates": 0}

    def run(argv, *args, **kwargs):
        if argv[1] == "create":
            state["creates"] += 1
            if state["creates"] <= n_failures:
                raise subprocess.CalledProcessError(1, "enroot", output=b"", stderr=b"boom")
        return MagicMock(stdout="", returncode=0)

    return run


def test_create_is_retried_then_succeeds():
    with patch("minisweagent.environments.enroot.subprocess.run") as m:
        m.side_effect = _failing_create(1)
        env = EnrootEnvironment(image="/img.sqsh", explicit_shell=False)

    assert [c.args[0][1] for c in m.call_args_list] == ["create", "remove", "create"]
    assert env.container_name


def test_create_raises_after_exhausting_retries():
    with patch("minisweagent.environments.enroot.subprocess.run") as m:
        m.side_effect = _failing_create(99)
        with pytest.raises(subprocess.CalledProcessError):
            EnrootEnvironment(image="/img.sqsh", create_retries=3, explicit_shell=False)

    assert [c.args[0][1] for c in m.call_args_list].count("create") == 3


# ------------------------------------------------------------------------ argv build


def test_execute_basic_argv(run_mock):
    env = _borrowed()

    env.execute({"command": "echo hi"})

    argv = _start_argv(run_mock)
    assert argv[:3] == ["enroot", "start", "--rw"]
    assert argv[-3] == "ctr"
    assert argv[-2] == "-c"
    assert argv[-1].endswith("echo hi")


def test_mounts_and_env_are_passed(run_mock):
    env = _borrowed(
        mounts=["/host:/workspace:none:bind,rw"],
        env={"FOO": "bar"},
    )

    env.execute({"command": "true"})

    argv = _start_argv(run_mock)
    assert "--mount" in argv
    assert argv[argv.index("--mount") + 1] == "/host:/workspace:none:bind,rw"
    assert "--env" in argv
    assert "FOO=bar" in argv


def test_forward_env_reads_host_and_skips_missing(run_mock):
    with patch.dict(os.environ, {"PRESENT": "yes"}, clear=False):
        os.environ.pop("DEFINITELY_ABSENT", None)
        env = _borrowed(forward_env=["PRESENT", "DEFINITELY_ABSENT"])
        env.execute({"command": "true"})

    argv = _start_argv(run_mock)
    assert "PRESENT=yes" in argv
    assert not any(a.startswith("DEFINITELY_ABSENT=") for a in argv)


def test_cwd_is_emulated_with_cd(run_mock):
    """enroot has no --pwd, so the cwd is prepended as a `cd`."""
    env = _borrowed(cwd="/app")

    env.execute({"command": "pwd"})

    assert "cd /app && pwd" in _start_argv(run_mock)[-1]


def test_cwd_argument_overrides_config(run_mock):
    env = _borrowed(cwd="/app")

    env.execute({"command": "pwd"}, cwd="/other dir")

    # shlex.quote protects paths containing spaces
    assert "cd '/other dir' && pwd" in _start_argv(run_mock)[-1]


def test_root_cwd_adds_no_cd(run_mock):
    env = _borrowed(cwd="/")

    env.execute({"command": "pwd"})

    assert "cd " not in _start_argv(run_mock)[-1]


def test_apply_image_env_prefix(run_mock):
    """The image's /etc/environment is applied, the way docker applies image ENV."""
    env = _borrowed()

    env.execute({"command": "go version"})

    command = _start_argv(run_mock)[-1]
    assert "/etc/environment" in command
    assert command.endswith("go version")


def test_apply_image_env_can_be_disabled(run_mock):
    env = _borrowed(apply_image_env=False)

    env.execute({"command": "go version"})

    assert "/etc/environment" not in _start_argv(run_mock)[-1]


def test_timeout_is_forwarded(run_mock):
    env = _borrowed(timeout=11)

    env.execute({"command": "true"})
    assert run_mock.call_args.kwargs["timeout"] == 11

    env.execute({"command": "true"}, timeout=99)
    assert run_mock.call_args.kwargs["timeout"] == 99


def test_start_new_session_is_set(run_mock):
    """Needed so a timeout kills enroot's whole helper process group."""
    _borrowed().execute({"command": "true"})

    assert run_mock.call_args.kwargs["start_new_session"] is True


# --------------------------------------------------------------------------- results


def test_successful_result_shape(run_mock):
    run_mock.return_value = MagicMock(stdout="out", returncode=0)

    result = _borrowed().execute({"command": "true"})

    assert result == {"output": "out", "returncode": 0, "exception_info": ""}


def test_nonzero_returncode_is_passed_through(run_mock):
    run_mock.return_value = MagicMock(stdout="", returncode=42)

    assert _borrowed().execute({"command": "exit 42"})["returncode"] == 42


def test_timeout_returns_structured_output_instead_of_raising(run_mock):
    run_mock.side_effect = subprocess.TimeoutExpired("enroot", 1, output=b"partial")

    result = _borrowed().execute({"command": "sleep 5"})

    assert result["returncode"] == -1
    assert "partial" in result["output"]
    assert "error occurred" in result["exception_info"]
    assert result["extra"]["exception_type"] == "TimeoutExpired"


def test_submitted_is_raised_on_sentinel(run_mock):
    run_mock.return_value = MagicMock(
        stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/x b/x\n", returncode=0
    )

    with pytest.raises(Submitted) as exc:
        _borrowed().execute({"command": "submit"})

    message = exc.value.messages[0]
    assert message["role"] == "exit"
    assert message["extra"]["exit_status"] == "Submitted"
    assert message["extra"]["submission"] == "diff --git a/x b/x\n"


def test_sentinel_with_nonzero_returncode_does_not_submit(run_mock):
    run_mock.return_value = MagicMock(stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", returncode=1)

    result = _borrowed().execute({"command": "submit"})

    assert result["returncode"] == 1


# ----------------------------------------------------------------------- protocol API


def test_serialize_reports_the_environment_type(run_mock):
    info = _borrowed().serialize()["info"]["config"]

    assert info["environment_type"] == "minisweagent.environments.enroot.EnrootEnvironment"
    assert info["environment"]["image"] == "/img.sqsh"


def test_get_template_vars_merges_extra_kwargs(run_mock):
    vars_ = _borrowed(cwd="/app").get_template_vars(extra="x")

    assert vars_["cwd"] == "/app"
    assert vars_["extra"] == "x"


# ------------------------------------------------------------- shell dispatch

# enroot renders the image's ENTRYPOINT/CMD into the rootfs's /etc/rc. Whether a
# shell is already interposed there decides how we must invoke `enroot start`;
# guessing wrong fails every command (127 one way, 126 the other).
RC_NO_ENTRYPOINT = """mkdir -p "/app" 2> /dev/null
cd "/app" && unset OLDPWD || exit 1

if [ $# -gt 0 ]; then
    exec  "$@"
else
    exec  'bash'
fi
"""

RC_BASH_ENTRYPOINT = """mkdir -p "/app" 2> /dev/null
cd "/app" && unset OLDPWD || exit 1

if [ $# -gt 0 ]; then
    exec '/bin/bash' "$@"
else
    exec '/bin/bash'
fi
"""


@pytest.fixture
def rootfs(tmp_path, monkeypatch):
    """A fake ENROOT_DATA_PATH; returns a writer for a container's /etc/rc."""
    monkeypatch.setenv("ENROOT_DATA_PATH", str(tmp_path))

    def write_rc(container: str, contents: str | None):
        etc = tmp_path / container / "etc"
        etc.mkdir(parents=True, exist_ok=True)
        if contents is not None:
            (etc / "rc").write_text(contents)

    return write_rc


def test_no_entrypoint_image_gets_an_explicit_shell(run_mock, rootfs):
    """`exec "$@"` execs our argv directly, so we must supply the shell."""
    rootfs("ctr", RC_NO_ENTRYPOINT)
    env = EnrootEnvironment(image="/img.sqsh", container_name="ctr")

    assert env._explicit_shell is True
    env.execute({"command": "true"})
    assert _start_argv(run_mock)[-3:-1] == ["bash", "-c"]


def test_bash_entrypoint_image_gets_no_extra_shell(run_mock, rootfs):
    """`exec '/bin/bash' "$@"` already interposes bash; another would be exit 126."""
    rootfs("ctr", RC_BASH_ENTRYPOINT)
    env = EnrootEnvironment(image="/img.sqsh", container_name="ctr")

    assert env._explicit_shell is False
    env.execute({"command": "true"})
    argv = _start_argv(run_mock)
    assert argv[-3] == "ctr"
    assert argv[-2] == "-c"


def test_explicit_shell_overrides_detection(run_mock, rootfs):
    rootfs("ctr", RC_BASH_ENTRYPOINT)

    assert EnrootEnvironment(image="/i", container_name="ctr", explicit_shell=True)._explicit_shell is True
    rootfs("ctr2", RC_NO_ENTRYPOINT)
    assert EnrootEnvironment(image="/i", container_name="ctr2", explicit_shell=False)._explicit_shell is False


def test_missing_rc_falls_back_to_explicit_shell(run_mock, rootfs):
    """A rootfs we cannot read must not silently produce 127 on every command."""
    rootfs("ctr", None)
    assert EnrootEnvironment(image="/i", container_name="ctr")._explicit_shell is True


def test_unrecognised_rc_falls_back_to_explicit_shell(run_mock, rootfs):
    rootfs("ctr", "# nothing that looks like a dispatch line\n")
    assert EnrootEnvironment(image="/i", container_name="ctr")._explicit_shell is True


# ------------------------------------------------------------------- real enroot runs


@pytest.mark.slow
@pytest.mark.skipif(not is_enroot_available(), reason="enroot not available")
def test_enroot_executable_is_discoverable():
    """Cheap guard that the runtime we build argv for actually exists."""
    result = subprocess.run(["enroot", "version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip()
