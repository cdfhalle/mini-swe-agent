import logging
import os
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge

# Loads the image's /etc/environment (the docker config Env that enroot stored
# there) into the command's environment. /etc/environment is pam_env-style
# KEY=VALUE -- values may contain spaces and quotes (PYTEST_ADDOPTS, NodeBB's
# SETUP JSON, ...) -- so each line is exported whole rather than `source`-ing the
# file (which would word-split them). Missing file / odd lines are ignored.
_APPLY_IMAGE_ENV = (
    '{ while IFS= read -r __e; do case "$__e" in ""|"#"*) : ;; *=*) export "$__e" ;; esac; '
    'done < /etc/environment; } 2>/dev/null; '
)


class EnrootEnvironmentConfig(BaseModel):
    image: str
    """Path to a pre-staged squashfs (.sqsh) image. (NOT a docker:// URI --
    enroot `create` needs a squashfs; do the `enroot import` in staging.)"""
    container_name: str = ""
    """Reuse an already-created enroot container by name. If empty, a unique
    container is created here and removed on cleanup."""
    cwd: str = "/"
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Host environment variables to forward into the container."""
    mounts: list[str] = []
    """Bind mounts, each "src:dst[:flags]", passed as `enroot start --mount`."""
    timeout: int = 30
    """Per-command timeout in seconds."""
    executable: str = os.getenv("MSWEA_ENROOT_EXECUTABLE", "enroot")
    """Path to the enroot executable."""
    create_retries: int = 3
    """Retries for `enroot create` (unpacking large images can occasionally flake)."""
    start_args: list[str] = ["--rw"]
    """Args to `enroot start`. `--rw` is REQUIRED so that file edits persist
    between commands (the analog of Singularity's `--writable`)."""
    shell: str = "bash"
    """Shell used to run the agent's command string. NOTE for SWE-Bench Pro:
    those images run bash by default (see Pro issue #6). enroot replaces the
    default command with whatever you pass to `start`, so an explicit
    `bash -c <cmd>` here normally sidesteps the Docker-style double-bash problem
    -- but verify against issue #6 for your specific images before a full run."""
    apply_image_env: bool = True
    """Apply the image's configured environment (PATH, GOPATH, etc.) before each
    command, the way Docker does automatically. enroot writes the docker image's
    Env to /etc/environment but does NOT reliably apply it under `start -c` (e.g.
    debian-based golang images get the host PATH, so `go` at /usr/local/go/bin is
    not found). We load it ourselves -- and to do so for BOTH generation and eval
    so the agent works in the same environment it is scored in."""


class EnrootEnvironment:
    def __init__(
        self,
        *,
        config_class: type = EnrootEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Enroot environment. See `EnrootEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        # If the caller supplied a name, assume the container already exists and
        # we are only borrowing it (do not create or remove it).
        self._owns_container = not self.config.container_name
        self.container_name = (
            self.config.container_name or f"minisweagent-{uuid.uuid4().hex[:8]}"
        )
        if self._owns_container:
            self._create_container()

    def _create_container(self) -> None:
        # Unpacks the squashfs into a writable rootfs under $ENROOT_DATA_PATH.
        # Point ENROOT_DATA_PATH at fast node-local storage (e.g. /tmp on NVMe),
        # NOT the shared FS, or parallel unpacks will thrash it.
        max_retries = self.config.create_retries
        for attempt in range(max_retries):
            try:
                subprocess.run(
                    [
                        self.config.executable,
                        "create",
                        "--name",
                        self.container_name,
                        self.config.image,
                    ],
                    check=True,
                    capture_output=True,
                )
                return
            except subprocess.CalledProcessError as e:
                # Remove a possibly half-created container before retrying.
                subprocess.run(
                    [self.config.executable, "remove", "-f", self.container_name],
                    capture_output=True,
                )
                self.logger.error(
                    f"enroot create failed for {self.config.image}, stdout: {getattr(e, 'stdout', b'')!r}, stderr: {getattr(e, 'stderr', b'')!r} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(
        self, action: dict, cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        """Execute a command in the enroot container and return the result as a dict."""
        command = action.get("command", "")

        # enroot has no `--pwd`; emulate it by prepending a cd. Requires a shell,
        # which the agent's commands need anyway.
        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            command = f"cd {shlex.quote(work_dir)} && {command}"

        # Apply the image's configured env (PATH etc.) first, like Docker does.
        if self.config.apply_image_env:
            command = _APPLY_IMAGE_ENV + command

        cmd = [self.config.executable, "start", *self.config.start_args]
        for mount in self.config.mounts:
            cmd += ["--mount", mount]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd += ["--env", f"{key}={value}"]
        for key, value in self.config.env.items():
            cmd += ["--env", f"{key}={value}"]
        # default: cmd += [self.container_name, self.config.shell, "-c", command]
        # swe-bench-pro images run bash by default so we should not specify it here (issue #6).
        cmd += [self.container_name, "-c", command]
        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # New session so a timeout can take down the whole process group
                # (enroot spawns helper processes).
                start_new_session=True,
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace")
                if isinstance(raw_output, bytes)
                else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        # Only remove containers we created ourselves.
        if getattr(self, "_owns_container", False):
            subprocess.run(
                [self.config.executable, "remove", "-f", self.container_name],
                capture_output=True,
            )

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
