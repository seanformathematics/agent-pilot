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
