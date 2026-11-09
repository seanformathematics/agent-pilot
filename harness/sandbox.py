"""Docker execution for agent code.

The rule: the harness never reads or writes anything in /work. /work belongs
entirely to the agent. Code goes into the container on stdin, and output
comes back through stdout/stderr pipes straight into files the harness owns,
in the run's outputs directory. Nothing in /work is ever trusted or touched.

One Sandbox per agent. Its working directory on the host is created empty at
start and mounted at /work. It is the only thing mounted: logs never are.

Container settings:
  * no network
  * memory, CPU and process-count limits from config (swap disabled)
  * all Linux capabilities dropped, no privilege escalation
  * read-only root filesystem; only /work and a small /tmp are writable
  * PID 1 is a tiny Python "reaper" that ignores SIGCHLD, so the kernel
    cleans up finished processes instead of leaving zombies that would use
    up the process limit

Each run_python call:
  1. writes the code to <output_dir>/<exec_id>.py on the host (the record)
  2. runs, inside the container, as the agent user:
         timeout -k 5 <T> python - ; kill -9 -1
     with the code on stdin, and stdout/stderr flowing directly into
     <output_dir>/<exec_id>.out and .err. The `kill -9 -1` runs as soon as
     Python finishes and kills everything else the agent user owns, so a
     detached process cannot hold the pipes open
  3. as a backstop, limits the docker exec call itself on the host, then
     kills leftover processes again from a separate docker exec
  4. reads each stream back from the harness's own files, capped (head and
     tail kept), so large output never sits in harness memory

Exit codes from `timeout`: 124 means the code was stopped by SIGTERM at the
time limit; 137 means it was SIGKILLed, either because it ignored SIGTERM for
5 more seconds, or because it hit the memory limit.
"""

import os
import re
import subprocess
import time
from pathlib import Path

DEFAULT_OUTPUT_CAP = 10_000  # per stream, counted in bytes
KILL_GRACE_S = 5  # seconds between SIGTERM and SIGKILL
HOST_MARGIN_S = 10  # host-side backstop on top of the code's own limit
EXEC_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

# PID 1 inside the container. Ignoring SIGCHLD tells the kernel to reap
# finished child processes automatically, so killed processes never pile up
# as zombies. `kill -9 -1` never touches PID 1.
REAPER = (
    "import signal\n"
    "signal.signal(signal.SIGCHLD, signal.SIG_IGN)\n"
    "while True:\n"
    "    signal.pause()\n"
)


class Sandbox:
    """A single agent's Docker container and working directory."""

    def __init__(
        self,
        image: str,
        work_dir: Path,
        memory: str = "1g",
        cpus: float = 1.0,
        pids_limit: int = 256,
        output_cap: int = DEFAULT_OUTPUT_CAP,
    ) -> None:
        self.image = image
        self.work_dir = Path(work_dir)
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.output_cap = output_cap
        self.container_id: str | None = None

    def start(self) -> None:
        """Create the empty working directory and start the container.

        The working directory must not already exist, so every agent starts
        with an empty /work and nothing carries over.
        """
        if self.container_id is not None:
            raise RuntimeError("sandbox already started")
        self.work_dir.mkdir(parents=True, exist_ok=False)
        command = [
            "docker", "run", "--detach",
            "--network", "none",
            "--memory", self.memory,
            "--memory-swap", self.memory,  # same value = no swap
            "--cpus", str(self.cpus),
            "--pids-limit", str(self.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:size=256m",
            "--mount", f"type=bind,source={self.work_dir.resolve()},target=/work",
            self.image,
            "python", "-c", REAPER,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        self.container_id = completed.stdout.strip()

    def stop(self) -> None:
        """Remove the container. Safe to call more than once."""
        if self.container_id is None:
            return
        subprocess.run(
            ["docker", "rm", "--force", self.container_id],
            capture_output=True,
            check=False,
        )
        self.container_id = None

    def run_python(self, code: str, timeout_s: int, exec_id: str, output_dir: Path) -> tuple[dict, dict]:
        """Execute ``code`` and return ``(result, details)``.

        ``output_dir`` is a harness-owned directory outside /work. The code
        and the full stdout/stderr are written there as <exec_id>.py/.out/.err.

        ``result`` is what the agent sees: exactly the keys exit_code, stdout,
        stderr, duration_s, timed_out, truncated, in that order.

        ``details`` is for the harness and the log only: how the process
        ended, and the paths and sizes of the three files.
        """
        if self.container_id is None:
            raise RuntimeError("sandbox not started")
        if not EXEC_ID_PATTERN.fullmatch(exec_id):
            raise ValueError(f"bad exec_id: {exec_id!r}")

        output_dir = Path(output_dir)
        script_path = output_dir / f"{exec_id}.py"
        stdout_path = output_dir / f"{exec_id}.out"
        stderr_path = output_dir / f"{exec_id}.err"
        with create_new_file(script_path) as f:
            f.write(code.encode("utf-8"))

        shell_line = (
            f"timeout -k {KILL_GRACE_S} {timeout_s} python -; "
            "rc=$?; kill -9 -1 2>/dev/null; exit $rc"
        )
        command = ["docker", "exec", "--interactive", self.container_id, "sh", "-c", shell_line]

        start = time.monotonic()
        with create_new_file(stdout_path) as out, create_new_file(stderr_path) as err:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err)
            try:
                process.communicate(input=code.encode("utf-8"), timeout=timeout_s + KILL_GRACE_S + HOST_MARGIN_S)
                exit_code = process.returncode
                host_timeout = False
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exit_code = -1
                host_timeout = True
        duration_s = round(time.monotonic() - start, 3)

        self._kill_leftover_processes()

        reason = classify_exit(exit_code, duration_s, timeout_s, host_timeout)
        stdout, stdout_truncated, stdout_bytes = read_capped(stdout_path, self.output_cap)
        stderr, stderr_truncated, stderr_bytes = read_capped(stderr_path, self.output_cap)

        result = {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "duration_s": duration_s,
            "timed_out": reason in ("timeout", "timeout_killed", "host_timeout"),
            "truncated": stdout_truncated or stderr_truncated,
        }
        details = {
            "end_reason": reason,
            "script_path": str(script_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_bytes": stdout_bytes,
            "stderr_bytes": stderr_bytes,
        }
        return result, details

    def _kill_leftover_processes(self) -> None:
        """Kill everything the agent user runs, except PID 1 (the reaper).

        Runs as the agent user: with all capabilities dropped, root inside
        the container is not allowed to signal another user's processes.
        """
        subprocess.run(
            ["docker", "exec", self.container_id, "sh", "-c", "kill -9 -1 2>/dev/null; true"],
            capture_output=True,
            check=False,
            timeout=HOST_MARGIN_S,
        )


def classify_exit(exit_code: int, duration_s: float, timeout_s: int, host_timeout: bool) -> str:
    """Name how an execution ended, for the log.

    Returns one of: "exited", "timeout", "timeout_killed", "killed", "host_timeout".
    "killed" means SIGKILL before the time limit, most likely the memory limit.
    """
    if host_timeout:
        return "host_timeout"
    if exit_code == 124:
        return "timeout"
    if exit_code == 137:
        return "timeout_killed" if duration_s >= timeout_s else "killed"
    return "exited"


def create_new_file(path: Path):
    """Open a brand-new file for binary writing.

    Fails if anything already exists at ``path``, including a symlink. A
    second layer of protection: the harness only writes outside /work, but
    it still never follows a link or overwrites a file.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    return os.fdopen(os.open(path, flags, 0o644), "wb")


def read_capped(path: Path, cap: int) -> tuple[str, bool, int]:
    """Read at most about ``cap`` bytes of a file: the head and the tail.

    Returns ``(text, truncated, total_bytes)``. Never loads the whole file when
    it is larger than the cap. A missing file reads as empty. Refuses to follow
    a symlink.
    """
    path = Path(path)
    if not os.path.lexists(path):
        return "", False, 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        total_bytes = os.fstat(f.fileno()).st_size
        if total_bytes <= cap:
            return f.read().decode("utf-8", errors="replace"), False, total_bytes
        half = cap // 2
        head = f.read(half)
        f.seek(total_bytes - half)
        tail = f.read(half)
    omitted = total_bytes - 2 * half
    note = f"\n[... truncated: {omitted} bytes omitted from the middle ...]\n"
    text = head.decode("utf-8", errors="replace") + note + tail.decode("utf-8", errors="replace")
    return text, True, total_bytes


def image_id(image: str) -> str:
    """The full content ID of a local Docker image, for the run log."""
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()
