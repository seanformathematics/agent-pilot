"""Sandbox helpers that need no Docker: output capping and exit classification."""

from harness.sandbox import classify_exit, read_capped


def test_small_file_read_whole(tmp_path):
    p = tmp_path / "a.out"
    p.write_text("hello\n")
    assert read_capped(p, 100) == ("hello\n", False, 6)


def test_file_exactly_at_cap_not_truncated(tmp_path):
    p = tmp_path / "a.out"
    p.write_bytes(b"x" * 100)
    text, truncated, total = read_capped(p, 100)
    assert not truncated and total == 100 and text == "x" * 100


def test_large_file_keeps_head_and_tail(tmp_path):
    p = tmp_path / "a.out"
    p.write_bytes(b"H" * 1000 + b"m" * 100_000 + b"T" * 1000)
    text, truncated, total = read_capped(p, 1000)
    assert truncated
    assert total == 102_000
    assert text.startswith("H" * 500)
    assert text.endswith("T" * 500)
    assert "truncated" in text
    assert len(text) < 1000 + 200  # cap plus a short note


def test_missing_file_reads_empty(tmp_path):
    assert read_capped(tmp_path / "nope.out", 100) == ("", False, 0)


def test_classify_exit():
    assert classify_exit(0, 0.1, 10, False) == "exited"
    assert classify_exit(1, 0.1, 10, False) == "exited"
    assert classify_exit(124, 10.0, 10, False) == "timeout"
    assert classify_exit(137, 15.0, 10, False) == "timeout_killed"
    assert classify_exit(137, 2.0, 10, False) == "killed"
    assert classify_exit(-1, 45.0, 10, True) == "host_timeout"


# ----- archive inspection (no extraction) --------------------------------

import io
import os
import tarfile

from harness.sandbox import inspect_archive


def _add(tar, name, kind=tarfile.REGTYPE, data=b"", linkname="", mode=0o644):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = len(data) if kind == tarfile.REGTYPE else 0
    info.linkname = linkname
    info.mode = mode
    tar.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)


def test_inspect_flags_risky_members_and_extracts_nothing(tmp_path):
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        _add(tar, "./ok.txt", data=b"fine")
        _add(tar, "./link", tarfile.SYMTYPE, linkname="/etc/passwd")
        _add(tar, "../escape.txt", data=b"x")
        _add(tar, "/abs.txt", data=b"x")
        _add(tar, "./pipe", tarfile.FIFOTYPE)
        _add(tar, "./suid", data=b"x", mode=0o4755)
    before = sorted(os.listdir(tmp_path))

    summary = inspect_archive(archive)

    assert sorted(os.listdir(tmp_path)) == before  # nothing extracted
    assert summary["complete"] is True
    assert summary["members"] == 6
    reasons = {f["name"]: f["reasons"] for f in summary["flagged"]}
    assert "ok.txt" not in str(reasons)
    assert reasons["./link"] == ["link"]
    assert "'..' in path" in reasons["../escape.txt"]
    assert "absolute path" in reasons["/abs.txt"]
    assert reasons["./pipe"] == ["special file"]
    assert reasons["./suid"] == ["setuid/setgid bit"]
    link = next(f for f in summary["flagged"] if f["name"] == "./link")
    assert link["target"] == "/etc/passwd"


def test_inspect_reports_truncated_archive(tmp_path):
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        _add(tar, "./one.txt", data=b"1" * 5000)
        _add(tar, "./two.txt", data=b"2" * 5000)
    data = archive.read_bytes()
    archive.write_bytes(data[:7000])  # cut inside the second file

    summary = inspect_archive(archive)
    assert summary["complete"] is False
    assert summary["members"] >= 1


# ----- the cleanup kill: when the container does not answer -------------------

import subprocess  # noqa: E402

from harness import sandbox as sandbox_module  # noqa: E402


def box_with_container():
    box = sandbox_module.Sandbox("image", 1024, 64, 64, 64, 1, 64, 1000, 1000)
    box.container_id = "some-container"
    return box


def test_cleanup_kill_reports_a_container_that_does_not_answer(monkeypatch):
    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(sandbox_module.subprocess, "run", hangs)
    assert box_with_container()._kill_leftover_processes() is False


def test_cleanup_kill_reports_docker_failing_to_run_it(monkeypatch):
    monkeypatch.setattr(sandbox_module.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1))
    assert box_with_container()._kill_leftover_processes() is False


def test_cleanup_kill_succeeds(monkeypatch):
    monkeypatch.setattr(sandbox_module.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 0))
    assert box_with_container()._kill_leftover_processes() is True
