"""The dummy logger matches the real logger's interface and writes readable JSONL."""

import json

import pytest

from tests.fixtures.dummy_logger import GENESIS, DummyLogger


def test_append_writes_one_line_per_entry(tmp_path):
    log = DummyLogger(tmp_path / "log.jsonl")
    log.append("agent-0", "task", {"text": "sum these"})
    log.append("agent-0", "tool_call", {"name": "run_python"})
    log.close()

    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines]
    assert [e["seq"] for e in entries] == [0, 1]
    assert entries[0]["prev_hash"] == GENESIS
    assert entries[1]["prev_hash"] == entries[0]["hash"]


def test_refuses_to_overwrite_existing_log(tmp_path):
    (tmp_path / "log.jsonl").write_text("existing\n")
    with pytest.raises(FileExistsError):
        DummyLogger(tmp_path / "log.jsonl")


def test_close_returns_last_hash(tmp_path):
    log = DummyLogger(tmp_path / "log.jsonl")
    h = log.append("agent-0", "task", {})
    assert log.last_hash == h
    assert log.close() == h
