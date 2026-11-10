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


# ----- the launcher: agent code behaves like a normal script -------------

def test_traceback_shows_source_lines_and_no_launcher_frames(make_sandbox):
    code = ("def magnetisation(spins):\n"
            "    return sum(spins) / len(spins)\n"
            "\n"
            "print(magnetisation([]))\n")
    result, _ = run(make_sandbox(), code, exec_id="tb-check")
    assert result["exit_code"] == 1
    stderr = result["stderr"]
    assert stderr.startswith("Traceback (most recent call last):\n")
    assert 'File "/tmp/run_python/tb-check.py", line 4, in <module>' in stderr
    assert "    return sum(spins) / len(spins)" in stderr      # the source line
    assert "~~~~~~~~~~^~~~~~~~~~~~" in stderr                     # the pointer
    assert "launcher" not in stderr and "runpy" not in stderr
    assert stderr.rstrip().endswith("ZeroDivisionError: division by zero")


def test_syntax_error_reported_like_a_script(make_sandbox):
    result, _ = run(make_sandbox(), "for i in range(3)\n    print(i)\n", exec_id="syn")
    assert result["exit_code"] == 1
    assert 'File "/tmp/run_python/syn.py", line 1' in result["stderr"]
    assert "SyntaxError: expected ':'" in result["stderr"]
    assert "launcher" not in result["stderr"]


def test_warnings_show_their_source_line(make_sandbox):
    code = "import warnings\nwarnings.warn('watch out')\n"
    result, _ = run(make_sandbox(), code)
    assert "UserWarning: watch out" in result["stderr"]
    assert "warnings.warn('watch out')" in result["stderr"]


def test_multiprocessing_pool_works(make_sandbox):
    code = (
        "import multiprocessing as mp\n"
        "def square(x):\n"
        "    return x * x\n"
        "if __name__ == '__main__':\n"
        "    with mp.Pool(2) as pool:\n"
        "        print(pool.map(square, range(5)))\n"
    )
    result, _ = run(make_sandbox(), code)
    assert result["exit_code"] == 0, result["stderr"]
    assert result["stdout"] == "[0, 1, 4, 9, 16]\n"


def test_script_basics(make_sandbox):
    code = ("import sys\n"
            "print(__name__, __file__, sys.argv[0])\n"
            "sys.exit(3)\n")
    result, _ = run(make_sandbox(), code, exec_id="basics")
    assert result["stdout"] == "__main__ /tmp/run_python/basics.py /tmp/run_python/basics.py\n"
    assert result["exit_code"] == 3


def test_launcher_cannot_be_modified_by_agent(make_sandbox):
    result, _ = run(make_sandbox(), "open('/opt/harness/launcher.py', 'a').write('# changed')")
    assert result["exit_code"] != 0


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
    box = make_sandbox(memory_mb=256)
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


# ----- workspace archive at the end of an agent -------------------------

import tarfile

from harness.sandbox import inspect_archive


def test_archive_holds_agent_files(make_sandbox):
    box = make_sandbox()
    run(box, "import os; os.makedirs('/work/sub'); open('/work/sub/result.txt','w').write('42')")
    info = box.archive_workspace(box.test_output_dir / "ws.tar", max_bytes=10 * 1024 * 1024)
    assert info["truncated"] is False and info["exit_code"] == 0
    summary = inspect_archive(box.test_output_dir / "ws.tar")
    assert summary["complete"] is True
    with tarfile.open(box.test_output_dir / "ws.tar") as tar:  # reading, not extracting
        assert tar.extractfile("./sub/result.txt").read() == b"42"


def test_archive_stores_links_not_host_files(make_sandbox, tmp_path):
    host_secret = tmp_path / "host_secret.txt"
    host_secret.write_text("HOST-ONLY-CONTENT")
    box = make_sandbox()
    run(box, f"import os; os.symlink({str(host_secret)!r}, '/work/sneaky')")
    box.archive_workspace(box.test_output_dir / "ws.tar", max_bytes=10 * 1024 * 1024)
    assert b"HOST-ONLY-CONTENT" not in (box.test_output_dir / "ws.tar").read_bytes()
    flagged = inspect_archive(box.test_output_dir / "ws.tar")["flagged"]
    assert flagged == [{"name": "./sneaky", "type": "symlink", "reasons": ["link"], "target": str(host_secret)}]


def test_archive_size_cap(make_sandbox):
    box = make_sandbox()
    run(box, "import os; open('/work/big.bin','wb').write(os.urandom(3 * 1024 * 1024))")
    info = box.archive_workspace(box.test_output_dir / "ws.tar", max_bytes=1024 * 1024)
    assert info["truncated"] is True
    assert os.path.getsize(box.test_output_dir / "ws.tar") == 1024 * 1024
    assert inspect_archive(box.test_output_dir / "ws.tar")["complete"] is False


def test_archive_cap_stops_huge_sparse_file_quickly(make_sandbox):
    # A 20 GB sparse file costs the agent nothing but would stream 20 GB of zeros.
    box = make_sandbox()
    run(box, "f = open('/work/huge.bin','wb'); f.truncate(20 * 1024**3); f.close()")
    import time
    start = time.monotonic()
    info = box.archive_workspace(box.test_output_dir / "ws.tar", max_bytes=1024 * 1024)
    assert info["truncated"] is True
    assert time.monotonic() - start < 30


def test_locked_workspace_is_still_cleared_and_removed(make_sandbox):
    box = make_sandbox()
    run(box, "import os; os.makedirs('/work/a/b'); open('/work/a/b/f','w').write('x');"
             " os.chmod('/work/a/b', 0); os.chmod('/work/a', 0)")
    assert box.clear_workspace() is True
    box.stop()
    assert box.remove_work_dir() is True
    assert not box.work_dir.exists()


def test_stored_output_cap_keeps_true_tail_for_agent(make_sandbox):
    box = make_sandbox(output_cap=1000, stored_output_cap=100_000)
    result, details = run(box, "print('H' * 600 + 'x' * 1_000_000 + 'END-OF-OUTPUT')")
    assert os.path.getsize(details["stdout_path"]) == 100_000
    assert details["stdout_bytes"] == 1_000_614
    assert details["stdout_stored_bytes"] == 100_000
    assert details["storage_truncated"] is True
    assert result["stdout"].startswith("H" * 500)
    assert result["stdout"].endswith("END-OF-OUTPUT\n")  # the real end, past the saved part


def test_stored_cap_smaller_than_agent_cap_refused(tmp_path):
    with pytest.raises(ValueError):
        Sandbox(IMAGE, tmp_path / "w", output_cap=1000, stored_output_cap=500)


def test_describe_image():
    facts = Sandbox.describe_image(IMAGE)
    assert facts["image_id"].startswith("sha256:")
    assert facts["python_version"].startswith("3.14")
    assert any(line.startswith("numpy==") for line in facts["packages"])


def test_api_key_never_enters_container(make_sandbox, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-must-not-leak")
    box = make_sandbox()
    code = (
        "import os, pathlib\n"
        "leaks = [k for k, v in os.environ.items() if 'ANTHROPIC' in k or 'sk-ant' in v]\n"
        "print(leaks)\n"
    )
    result, _ = run(box, code)
    assert result["stdout"] == "[]\n"
    inspect = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Config.Env}}", box.container_id],
        capture_output=True, text=True, check=True,
    )
    assert "sk-ant" not in inspect.stdout and "ANTHROPIC" not in inspect.stdout


# ----- listing of writable places after each call ------------------------

def read_listing(details):
    with open(details["listing_path"]) as f:
        return [json.loads(line) for line in f]


def test_listing_covers_work_tmp_and_shm_without_contents(make_sandbox):
    box = make_sandbox()
    code = (
        "open('/work/a.txt', 'w').write('SECRET-CONTENT-A')\n"
        "open('/tmp/b.bin', 'wb').write(b'x' * 1234)\n"
        "open('/dev/shm/c', 'w').write('SECRET-CONTENT-C')\n"
    )
    _, details = run(box, code, exec_id="listing")
    listing = read_listing(details)
    by_path = {e["path"]: e for e in listing}
    assert by_path["/work/a.txt"]["bytes"] == 16 and by_path["/work/a.txt"]["dir"] == "/work"
    assert by_path["/tmp/b.bin"]["bytes"] == 1234 and by_path["/tmp/b.bin"]["type"] == "f"
    assert by_path["/dev/shm/c"]["dir"] == "/dev/shm"
    assert all(e["mtime"] > 1.7e9 for e in listing)
    with open(details["listing_path"]) as f:
        assert "SECRET-CONTENT" not in f.read()
    # /tmp also holds the launcher's copy of the code it ran
    assert by_path["/tmp/run_python/listing.py"]["bytes"] == len(code)
    summary = details["filesystem"]["dirs"]
    assert summary["/tmp"] == {"entries": 3, "files": 2, "file_bytes": 1234 + len(code)}
    assert details["filesystem"]["listing_truncated"] is False


def test_listing_shows_state_persisting_between_calls(make_sandbox):
    box = make_sandbox()
    run(box, "open('/tmp/kept.txt', 'w').write('hidden state')")
    _, details = run(box, "print('a later call that writes nothing')")
    assert "/tmp/kept.txt" in {e["path"] for e in read_listing(details)}


def test_listing_handles_awkward_file_names(make_sandbox):
    box = make_sandbox()
    _, details = run(box, "open('/work/tab\\tand\\nnewline.txt', 'w').write('x')")
    assert "/work/tab\tand\nnewline.txt" in {e["path"] for e in read_listing(details)}


def test_listing_records_symlinks_without_following(make_sandbox):
    box = make_sandbox()
    _, details = run(box, "import os; os.symlink('/etc/passwd', '/work/link')")
    entry = next(e for e in read_listing(details) if e["path"] == "/work/link")
    assert entry["type"] == "l"


def test_listing_is_capped(make_sandbox, monkeypatch):
    import harness.sandbox as sandbox_module
    monkeypatch.setattr(sandbox_module, "LISTING_MAX_BYTES", 2000)
    box = make_sandbox()
    _, details = run(box, "import os\nfor i in range(500): open(f'/work/file_{i:04d}.txt', 'w').close()")
    assert details["filesystem"]["listing_truncated"] is True
    listing = read_listing(details)
    assert 0 < len(listing) < 500
    assert all(e["path"].startswith("/work/file_") for e in listing)  # no half-read entries


def test_tmp_and_shm_sizes_come_from_settings(make_sandbox):
    box = make_sandbox(memory_mb=512, tmp_mb=100, shm_mb=50)
    code = (
        "import os\n"
        "for d in ('/tmp', '/dev/shm'):\n"
        "    s = os.statvfs(d)\n"
        "    print(d, s.f_blocks * s.f_frsize // (1024 * 1024))\n"
    )
    result, _ = run(box, code)
    assert result["stdout"] == "/tmp 100\n/dev/shm 50\n"
