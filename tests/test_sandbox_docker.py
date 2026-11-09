"""Sandbox behaviour in a real container. Needs Docker and the sandbox image.

Build the image with:  docker build -t residency-sandbox:py3.14 docker/
"""

import json
import os
import subprocess
import uuid

import pytest

from harness.sandbox import Sandbox
from harness.tools import RUN_PYTHON_RESULT_KEYS

IMAGE = "residency-sandbox:py3.14"


def _image_available() -> bool:
    try:
        return subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True
        ).returncode == 0
    except FileNotFoundError:
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _image_available(), reason=f"Docker image {IMAGE} not available"),
]


@pytest.fixture
def make_sandbox(tmp_path):
    """Start sandboxes whose outputs go to tmp_path/outputs/<name>, never /work."""
    started = []

    def make(name="agent", **kwargs):
        box = Sandbox(IMAGE, tmp_path / "work" / name, **kwargs)
        box.start()
        box.test_output_dir = tmp_path / "outputs" / name
        box.test_output_dir.mkdir(parents=True)
        started.append(box)
        return box

    yield make
    for box in started:
        box.stop()


def run(box, code, timeout_s=30, exec_id=None):
    exec_id = exec_id or f"exec-{uuid.uuid4().hex[:8]}"
    return box.run_python(code, timeout_s, exec_id, box.test_output_dir)


SEARCH_FILESYSTEM = (
    "import os\n"
    "hits = []\n"
    "for root, dirs, files in os.walk('/'):\n"
    "    dirs[:] = [d for d in dirs if os.path.join(root, d) not in ('/proc', '/sys')]\n"
    "    if NAME in files: hits.append(root)\n"
    "print(hits)\n"
)


# ----- basic behaviour ---------------------------------------------------

def test_hello_returns_exact_keys(make_sandbox):
    result, details = run(make_sandbox(), "print('hi')")
    assert tuple(result) == RUN_PYTHON_RESULT_KEYS
    assert result["exit_code"] == 0
    assert result["stdout"] == "hi\n"
    assert result["stderr"] == ""
    assert result["timed_out"] is False and result["truncated"] is False
    assert details["end_reason"] == "exited"


def test_error_exit_and_stderr(make_sandbox):
    result, _ = run(make_sandbox(), "import sys; print('bad', file=sys.stderr); sys.exit(3)")
    assert result["exit_code"] == 3
    assert result["stderr"] == "bad\n"


def test_traceback_names_the_line(make_sandbox):
    result, _ = run(make_sandbox(), "x = 1\ny = x / 0\n")
    assert result["exit_code"] == 1
    assert 'File "<stdin>", line 2' in result["stderr"]
    assert "ZeroDivisionError" in result["stderr"]


def test_files_persist_between_calls(make_sandbox):
    box = make_sandbox()
    run(box, "open('/work/state.txt', 'w').write('kept')")
    result, _ = run(box, "print(open('/work/state.txt').read())")
    assert result["stdout"] == "kept\n"


# ----- the harness stays out of /work ------------------------------------

def test_harness_creates_no_files_in_work(make_sandbox):
    box = make_sandbox()
    result, details = run(box, "print('hello')")
    assert result["stdout"] == "hello\n"
    assert os.listdir(box.work_dir) == []
    for key in ("script_path", "stdout_path", "stderr_path"):
        assert os.path.dirname(details[key]) == str(box.test_output_dir)


def test_record_files_hold_code_and_full_output(make_sandbox):
    box = make_sandbox()
    code = "print('a'); import sys; print('b', file=sys.stderr)"
    _, details = run(box, code)
    with open(details["script_path"]) as f:
        assert f.read() == code
    with open(details["stdout_path"]) as f:
        assert f.read() == "a\n"
    with open(details["stderr_path"]) as f:
        assert f.read() == "b\n"


def test_symlinks_planted_in_work_have_no_effect(make_sandbox, tmp_path):
    # Worst case: the agent knows a host path and plants links to it at every
    # name the old design used. The harness no longer looks in /work at all.
    host_file = tmp_path / "host_file.txt"
    host_file.write_text("ORIGINAL")
    box = make_sandbox()
    plant = (
        "import os\n"
        "for suffix in ('.py', '.out', '.err'):\n"
        f"    os.symlink({str(host_file)!r}, '/work/next' + suffix)\n"
    )
    run(box, plant)
    result, _ = run(box, "print('fresh')", exec_id="next")
    assert result["stdout"] == "fresh\n"
    assert host_file.read_text() == "ORIGINAL"


def test_output_files_are_never_overwritten(make_sandbox):
    # Second layer: the harness refuses to reuse a name in its own directory.
    box = make_sandbox()
    run(box, "print(1)", exec_id="same")
    with pytest.raises(FileExistsError):
        run(box, "print(2)", exec_id="same")


# ----- time limits and leftover processes --------------------------------

def test_timeout_stops_code(make_sandbox):
    result, details = run(make_sandbox(), "import time; time.sleep(30)", timeout_s=1)
    assert result["exit_code"] == 124
    assert result["timed_out"] is True
    assert details["end_reason"] == "timeout"
    assert result["duration_s"] < 10


def test_code_ignoring_sigterm_is_killed(make_sandbox):
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    result, details = run(make_sandbox(), code, timeout_s=1)
    assert result["exit_code"] == 137
    assert result["timed_out"] is True
    assert details["end_reason"] == "timeout_killed"


def test_background_processes_do_not_survive(make_sandbox):
    box = make_sandbox()
    run(box, "import subprocess; subprocess.Popen(['sleep', '300'], start_new_session=True,"
             " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)")
    code = (
        "import os\n"
        "cmds = [open(f'/proc/{p}/cmdline','rb').read() for p in os.listdir('/proc') if p.isdigit()]\n"
        "print(any(b'300' in c for c in cmds))\n"
    )
    result, _ = run(box, code)
    assert result["stdout"] == "False\n"


def test_detached_process_holding_stdout_does_not_hang(make_sandbox):
    # The detached sleep inherits stdout and stderr, so it holds the pipes open.
    box = make_sandbox()
    code = (
        "import subprocess\n"
        "subprocess.Popen(['sleep', '300'], start_new_session=True)\n"
        "print('main finished')\n"
    )
    result, details = run(box, code, timeout_s=30)
    assert result["exit_code"] == 0
    assert result["stdout"] == "main finished\n"
    assert details["end_reason"] == "exited"
    assert result["duration_s"] < 10


def test_killed_processes_do_not_exhaust_process_limit(make_sandbox):
    # Each call leaves an orphan that the harness kills. Without a reaper as
    # PID 1 these become zombies and use up the 32-process limit.
    box = make_sandbox(pids_limit=32)
    leave_orphan = ("import subprocess; subprocess.Popen(['sleep', '300'], start_new_session=True,"
                    " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)")
    for _ in range(40):
        run(box, leave_orphan)
    code = (
        "import os, subprocess\n"
        "zombies = sum(1 for p in os.listdir('/proc') if p.isdigit()"
        " and 'State:\\tZ' in open(f'/proc/{p}/status').read())\n"
        "subprocess.run(['true'], check=True)\n"
        "print(zombies)\n"
    )
    result, _ = run(box, code)
    assert result["exit_code"] == 0
    assert result["stdout"] == "0\n"


# ----- resource limits ---------------------------------------------------

def test_memory_limit_kills_code(make_sandbox):
    box = make_sandbox(memory="256m")
    result, details = run(box, "x = b'a' * (800 * 1024 * 1024); print(len(x))")
    assert result["exit_code"] == 137
    assert result["timed_out"] is False
    assert details["end_reason"] == "killed"


def test_streamed_output_is_capped_but_kept_in_full(make_sandbox):
    box = make_sandbox(output_cap=1000)
    result, details = run(box, "print('H' * 600 + 'x' * 50_000 + 'T' * 600)")
    assert result["truncated"] is True
    assert len(result["stdout"]) < 1200
    assert result["stdout"].startswith("H" * 500)
    assert result["stdout"].rstrip("\n").endswith("T" * 499)
    assert details["stdout_bytes"] == 51_201
    assert os.path.getsize(details["stdout_path"]) == 51_201


def test_no_network(make_sandbox):
    code = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
    result, _ = run(make_sandbox(), code)
    assert result["exit_code"] != 0


def test_root_filesystem_is_read_only(make_sandbox):
    result, _ = run(make_sandbox(), "open('/etc/planted', 'w').write('x')")
    assert result["exit_code"] != 0
    assert "Read-only file system" in result["stderr"]


# ----- isolation ---------------------------------------------------------

def test_work_dir_must_be_fresh(tmp_path):
    existing = tmp_path / "already-here"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        Sandbox(IMAGE, existing).start()


def test_two_agents_cannot_see_each_others_files(make_sandbox):
    a = make_sandbox("agent-a")
    b = make_sandbox("agent-b")
    marker = f"secret-{uuid.uuid4().hex}.txt"
    result, _ = run(a, f"open('/work/{marker}', 'w').write('from a')")
    assert result["exit_code"] == 0
    result, _ = run(b, SEARCH_FILESYSTEM.replace("NAME", repr(marker)), timeout_s=60)
    assert result["exit_code"] == 0
    assert result["stdout"] == "[]\n"


def test_log_and_output_directories_not_reachable(make_sandbox, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_name = f"log-{uuid.uuid4().hex}.jsonl"
    (log_dir / log_name).write_text('{"kind": "secret"}\n')

    box = make_sandbox()
    _, details = run(box, "print('recorded')")
    output_name = os.path.basename(details["stdout_path"])

    for name in (log_name, output_name):
        result, _ = run(box, SEARCH_FILESYSTEM.replace("NAME", repr(name)), timeout_s=60)
        assert result["stdout"] == "[]\n"

    inspect = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Mounts}}", box.container_id],
        capture_output=True, text=True, check=True,
    )
    binds = [m for m in json.loads(inspect.stdout) if m["Type"] == "bind"]
    assert len(binds) == 1
    assert binds[0]["Destination"] == "/work"
    assert binds[0]["Source"].endswith("work/agent")


def test_stop_removes_container(make_sandbox):
    box = make_sandbox()
    cid = box.container_id
    box.stop()
    gone = subprocess.run(["docker", "inspect", cid], capture_output=True)
    assert gone.returncode != 0
