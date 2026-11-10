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
  * /tmp and /dev/shm sizes from config; both live in memory and count
    against the memory limit
  * all Linux capabilities dropped, no privilege escalation
  * read-only root filesystem; only /work and a small /tmp are writable
  * PID 1 is a tiny Python "reaper" that ignores SIGCHLD, so the kernel
    cleans up finished processes instead of leaving zombies that would use
    up the process limit

Each run_python call:
  1. writes the code to <output_dir>/<exec_id>.py on the host (the record)
  2. runs, inside the container, as the agent user:
         timeout -k 5 <T> python /opt/harness/launcher.py <exec_id> ; kill -9 -1
     with the code on stdin. The launcher (docker/launcher.py, baked into
     the image) saves the code as /tmp/run_python/<exec_id>.py and runs it
     as a normal script, so tracebacks show source lines and
     multiprocessing works. The `kill -9 -1` runs as soon as Python
     finishes and kills everything else the agent user owns, so a detached
     process cannot hold the pipes open
  3. streams stdout and stderr through pipes in small chunks: each stream is
     saved to <output_dir>/<exec_id>.out/.err up to stored_output_cap bytes,
     and a small rolling buffer keeps its last few kilobytes. Memory use is
     bounded no matter how much the code prints
  4. as a backstop, limits the docker exec call itself on the host, then
     kills leftover processes again from a separate docker exec
  5. builds what the agent sees from the saved head and the rolling tail,
     capped at output_cap (head and tail kept, truncation noted)
  6. lists every writable place the agent can keep things (/work, /tmp,
     /dev/shm): type, size, modification time and path, never contents.
     `find` runs inside the container and streams the listing out, so the
     host still never reads /work. Saved as <exec_id>.files.jsonl

When the agent has ended (see Harness.stop_agent):
  1. archive_workspace: `tar` runs inside the container, as the agent user,
     and streams a tar of /work out through a pipe into a harness-owned file,
     capped in size. The harness sees only the byte stream.
  2. inspect_archive: lists the archive's members without extracting
     anything, and flags links, absolute or "../" paths, special files and
     setuid bits. Nothing is ever extracted automatically.
  3. clear_workspace: deletes the contents of /work from inside the
     container, so the host never walks agent-made files.
  4. stop, then remove_work_dir: removes the now-empty directory on the host.

Exit codes from `timeout`: 124 means the code was stopped by SIGTERM at the
time limit; 137 means it was SIGKILLed, either because it ignored SIGTERM for
5 more seconds, or because it hit the memory limit.
"""

import json
import os
import re
import stat
import subprocess
import tarfile
import threading
import time
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT_CAP = 10_000  # per stream shown to the agent, counted in bytes
DEFAULT_STORED_OUTPUT_CAP = 100 * 1024 * 1024  # per stream saved to disk
STREAM_CHUNK_BYTES = 64 * 1024
KILL_GRACE_S = 5  # seconds between SIGTERM and SIGKILL
HOST_MARGIN_S = 10  # host-side backstop on top of the code's own limit
EXEC_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
LAUNCHER = "/opt/harness/launcher.py"  # in the image; see docker/launcher.py
ARCHIVE_CHUNK_BYTES = 1 << 20  # read the tar stream 1 MiB at a time
ARCHIVE_TIMEOUT_S = 300
MAX_FLAGGED_MEMBERS = 100  # how many flagged archive entries to list in the log
SCRATCH_DIRS = ("/work", "/tmp", "/dev/shm")  # everywhere the agent can write
LISTING_MAX_BYTES = 5 * 1024 * 1024  # raw `find` output read per listing
LISTING_TIMEOUT_S = 60

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
        memory_mb: int = 1024,
        tmp_mb: int = 256,
        shm_mb: int = 256,
        cpus: float = 1.0,
        pids_limit: int = 256,
        output_cap: int = DEFAULT_OUTPUT_CAP,
        stored_output_cap: int = DEFAULT_STORED_OUTPUT_CAP,
    ) -> None:
        if stored_output_cap < output_cap:
            raise ValueError("stored_output_cap must be at least output_cap")
        self.image = image
        self.work_dir = Path(work_dir)
        self.memory_mb = memory_mb
        self.tmp_mb = tmp_mb
        self.shm_mb = shm_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.output_cap = output_cap
        self.stored_output_cap = stored_output_cap
        self.container_id: str | None = None

    @staticmethod
    def describe_image(image: str) -> dict:
        """Facts about the sandbox image for the run log: ID, Python, packages."""
        def inside(*args: str) -> str:
            completed = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", image, *args],
                capture_output=True, text=True, check=True, timeout=120,
            )
            return completed.stdout.strip()

        return {
            "image": image,
            "image_id": image_id(image),
            "python_version": inside("python", "-c", "import sys; print(sys.version)"),
            "packages": inside("pip", "list", "--format=freeze").splitlines(),
        }

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
            "--memory", f"{self.memory_mb}m",
            "--memory-swap", f"{self.memory_mb}m",  # same value = no swap
            "--cpus", str(self.cpus),
            "--pids-limit", str(self.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", f"/tmp:size={self.tmp_mb}m",
            "--shm-size", f"{self.shm_mb}m",
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
            f"timeout -k {KILL_GRACE_S} {timeout_s} python {LAUNCHER} {exec_id}; "
            "rc=$?; kill -9 -1 2>/dev/null; exit $rc"
        )
        command = ["docker", "exec", "--interactive", self.container_id, "sh", "-c", shell_line]

        half = self.output_cap // 2
        out_stats, err_stats = {}, {}
        start = time.monotonic()
        with create_new_file(stdout_path) as out, create_new_file(stderr_path) as err:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            readers = [
                threading.Thread(target=drain_stream,
                                 args=(process.stdout, out, self.stored_output_cap, half, out_stats)),
                threading.Thread(target=drain_stream,
                                 args=(process.stderr, err, self.stored_output_cap, half, err_stats)),
            ]
            for reader in readers:
                reader.start()
            try:
                process.stdin.write(code.encode("utf-8"))
                process.stdin.close()
            except BrokenPipeError:
                pass  # the process ended before reading all of its input
            try:
                process.wait(timeout=timeout_s + KILL_GRACE_S + HOST_MARGIN_S)
                exit_code = process.returncode
                host_timeout = False
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                exit_code = -1
                host_timeout = True
            for reader in readers:
                reader.join()  # the pipes close once the docker client has exited
        duration_s = round(time.monotonic() - start, 3)

        self._kill_leftover_processes()

        reason = classify_exit(exit_code, duration_s, timeout_s, host_timeout)
        stdout, stdout_truncated = agent_view(stdout_path, out_stats, self.output_cap)
        stderr, stderr_truncated = agent_view(stderr_path, err_stats, self.output_cap)
        listing_path = output_dir / f"{exec_id}.files.jsonl"
        filesystem = self.list_scratch_dirs(listing_path)

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
            "stdout_bytes": out_stats["total"],
            "stderr_bytes": err_stats["total"],
            "stdout_stored_bytes": out_stats["stored"],
            "stderr_stored_bytes": err_stats["stored"],
            "storage_truncated": (out_stats["stored"] < out_stats["total"]
                                  or err_stats["stored"] < err_stats["total"]),
            "listing_path": str(listing_path),
            "filesystem": filesystem,
        }
        return result, details

    def list_scratch_dirs(self, listing_path: Path) -> dict:
        """Write a listing of /work, /tmp and /dev/shm to ``listing_path``.

        One JSON line per entry: dir, path, type (f file, d directory,
        l symlink, p pipe, s socket), bytes, mtime (seconds since 1970).
        Never contents. Returns a short summary for the log.
        """
        command = ["docker", "exec", self.container_id, "find", *SCRATCH_DIRS, "-mindepth", "1",
                   "-printf", "%y\\t%s\\t%T@\\t%p\\0"]
        raw, truncated, exit_code = read_command_output(command, LISTING_MAX_BYTES, LISTING_TIMEOUT_S)
        if truncated:
            raw = raw[: raw.rfind(b"\0") + 1]  # drop the last, partial entry

        summary = {d: {"entries": 0, "files": 0, "file_bytes": 0} for d in SCRATCH_DIRS}
        with create_new_file(listing_path) as f:
            for item in raw.split(b"\0"):
                parts = item.split(b"\t", 3)
                if len(parts) != 4:
                    continue
                kind, size, mtime, path = (part.decode("utf-8", errors="surrogateescape") for part in parts)
                top = next((d for d in SCRATCH_DIRS if path == d or path.startswith(d + "/")), None)
                record = {"dir": top, "path": path, "type": kind, "bytes": int(size), "mtime": float(mtime)}
                f.write((json.dumps(record) + "\n").encode("utf-8"))
                if top:
                    summary[top]["entries"] += 1
                    if kind == "f":
                        summary[top]["files"] += 1
                        summary[top]["file_bytes"] += int(size)
        return {"dirs": summary, "listing_truncated": truncated, "find_exit_code": exit_code}

    def archive_workspace(self, archive_path: Path, max_bytes: int) -> dict:
        """Stream a tar of /work into ``archive_path``, stopping at ``max_bytes``.

        Call only after the agent has ended. Leftover processes are killed
        first so nothing changes during the copy. tar's own error messages
        go to ``<archive_path>.err``.

        Returns bytes written, whether the cap or the time limit cut it short,
        tar's exit code and the start of its error output.
        """
        if self.container_id is None:
            raise RuntimeError("sandbox not started")
        self._kill_leftover_processes()

        archive_path = Path(archive_path)
        errors_path = archive_path.with_name(archive_path.name + ".err")
        command = ["docker", "exec", self.container_id,
                   "/usr/bin/tar", "--create", "--file=-", "--directory=/work", "."]

        bytes_written = 0
        truncated = False
        timer_fired = []

        def on_timeout():
            timer_fired.append(True)
            process.kill()

        with create_new_file(archive_path) as archive, create_new_file(errors_path) as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            timer = threading.Timer(ARCHIVE_TIMEOUT_S, on_timeout)
            timer.start()
            try:
                while chunk := process.stdout.read(ARCHIVE_CHUNK_BYTES):
                    room = max_bytes - bytes_written
                    if len(chunk) > room:
                        archive.write(chunk[:room])
                        bytes_written += room
                        truncated = True
                        process.kill()
                        break
                    archive.write(chunk)
                    bytes_written += len(chunk)
            finally:
                timer.cancel()
                process.stdout.close()
                exit_code = process.wait()

        self._kill_leftover_processes()  # a killed docker client can leave tar running
        tar_errors, _, _ = read_capped(errors_path, 2000)
        return {
            "archive_path": str(archive_path),
            "errors_path": str(errors_path),
            "bytes": bytes_written,
            "truncated": truncated,
            "timed_out": bool(timer_fired),
            "exit_code": exit_code,
            "tar_errors": tar_errors,
        }

    def clear_workspace(self) -> bool:
        """Delete everything in /work from inside the container.

        Runs as the agent user, who owns every file there. Permissions are
        opened up first, in case the agent locked a directory. Returns True
        if /work is empty afterwards.
        """
        if self.container_id is None:
            raise RuntimeError("sandbox not started")
        self._kill_leftover_processes()
        script = ("chmod -R u+rwX /work 2>/dev/null; "
                  "find /work -mindepth 1 -delete 2>/dev/null; "
                  'test -z "$(ls -A /work)"')
        completed = subprocess.run(
            ["docker", "exec", self.container_id, "sh", "-c", script],
            capture_output=True, check=False, timeout=ARCHIVE_TIMEOUT_S,
        )
        return completed.returncode == 0

    def remove_work_dir(self) -> bool:
        """Remove the host working directory if it is empty. Returns True if removed."""
        try:
            os.rmdir(self.work_dir)  # refuses to remove a non-empty directory
            return True
        except OSError:
            return False

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


def read_command_output(command: list[str], max_bytes: int, timeout_s: float) -> tuple[bytes, bool, int]:
    """Run a command and read at most ``max_bytes`` of its stdout.

    Returns ``(output, truncated, exit_code)``. Stops the command if it
    produces more, or runs longer than ``timeout_s``. Memory stays bounded.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    timer = threading.Timer(timeout_s, process.kill)
    timer.start()
    chunks = []
    received = 0
    truncated = False
    try:
        while chunk := process.stdout.read(STREAM_CHUNK_BYTES):
            room = max_bytes - received
            if len(chunk) > room:
                chunks.append(chunk[:room])
                truncated = True
                process.kill()
                break
            chunks.append(chunk)
            received += len(chunk)
    finally:
        timer.cancel()
        process.stdout.close()
        exit_code = process.wait()
    return b"".join(chunks), truncated, exit_code


def drain_stream(pipe, file, store_limit: int, tail_bytes: int, stats: dict) -> None:
    """Copy a pipe to ``file`` until it closes, saving at most ``store_limit`` bytes.

    Keeps reading after the limit so the process never blocks on a full
    pipe. Also keeps the last ``tail_bytes`` of the whole stream in memory.
    Fills ``stats`` with "total", "stored" and "tail".
    """
    total = 0
    stored = 0
    tail = bytearray()
    while chunk := pipe.read(STREAM_CHUNK_BYTES):
        total += len(chunk)
        if stored < store_limit:
            part = chunk[: store_limit - stored]
            file.write(part)
            stored += len(part)
        tail += chunk
        if len(tail) > tail_bytes:
            del tail[:-tail_bytes]
    stats.update(total=total, stored=stored, tail=bytes(tail))


def agent_view(saved_path: Path, stats: dict, cap: int) -> tuple[str, bool]:
    """What the agent sees of one stream: all of it, or its head and tail.

    The head comes from the saved file and the tail from the rolling buffer,
    so the true end of the output is shown even if saving stopped early.
    """
    if stats["total"] <= cap:
        text, _, _ = read_capped(saved_path, cap)
        return text, False
    half = cap // 2
    fd = os.open(saved_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        head = f.read(half)
    tail = stats["tail"][-half:]
    omitted = stats["total"] - len(head) - len(tail)
    note = f"\n[... truncated: {omitted} bytes omitted from the middle ...]\n"
    text = head.decode("utf-8", errors="replace") + note + tail.decode("utf-8", errors="replace")
    return text, True


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


def inspect_archive(path: Path) -> dict:
    """Summarise a tar archive WITHOUT extracting anything.

    Reads only the member headers. Flags entries that would be risky to
    extract: symlinks and hard links, absolute paths, paths containing "..",
    device files and FIFOs, and setuid/setgid bits. A truncated archive is
    reported as incomplete; the entries before the cut are still listed.
    """
    members = 0
    total_bytes = 0
    types = Counter()
    flagged = []
    complete = True
    error = None
    try:
        with tarfile.open(path, mode="r:") as tar:
            for member in tar:
                members += 1
                total_bytes += member.size
                types[member_type(member)] += 1
                reasons = risk_reasons(member)
                if reasons:
                    entry = {"name": member.name, "type": member_type(member), "reasons": reasons}
                    if member.issym() or member.islnk():
                        entry["target"] = member.linkname
                    flagged.append(entry)
    except (tarfile.TarError, EOFError) as exc:
        complete = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "complete": complete,
        "error": error,
        "members": members,
        "total_member_bytes": total_bytes,
        "types": dict(types),
        "flagged_count": len(flagged),
        "flagged": flagged[:MAX_FLAGGED_MEMBERS],
    }


def member_type(member: tarfile.TarInfo) -> str:
    if member.isfile():
        return "file"
    if member.isdir():
        return "dir"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    if member.isfifo():
        return "fifo"
    if member.ischr() or member.isblk():
        return "device"
    return "other"


def risk_reasons(member: tarfile.TarInfo) -> list[str]:
    reasons = []
    if member.name.startswith("/"):
        reasons.append("absolute path")
    if ".." in Path(member.name).parts:
        reasons.append("'..' in path")
    if member.issym() or member.islnk():
        reasons.append("link")
    if member.isfifo() or member.ischr() or member.isblk():
        reasons.append("special file")
    if member.mode & (stat.S_ISUID | stat.S_ISGID):
        reasons.append("setuid/setgid bit")
    return reasons


def image_id(image: str) -> str:
    """The full content ID of a local Docker image, for the run log."""
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()
