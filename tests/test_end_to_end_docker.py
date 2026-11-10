"""A full single-agent run of the toy experiment: fake model, real container,
real logger and real verifier."""

import json
import shutil
import subprocess
import tarfile

import pytest

from harness.logger import Logger
from harness.run import run_experiment
from harness.sandbox import Sandbox
from tests.conftest import TOY_CONFIG, FakeClient, tool_call
from harness.verify import verify

IMAGE = "residency-sandbox:py3.14"


def _image_available() -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _image_available(), reason=f"Docker image {IMAGE} not available"),
]


def test_toy_run_end_to_end(tmp_path):
    client = FakeClient([
        tool_call("run_python", {"code": (
            "total = sum([17, -4, 256, 1031, 9, -88, 42, 7, 300, -15])\n"
            "open('/work/answer.txt', 'w').write(str(total))\n"
            "print(total)\n")}),
        tool_call("submit_result", {"total": 1555, "method_notes": "sum() in Python."}),
    ])
    record = run_experiment(TOY_CONFIG, client, Logger, sandbox_factory=Sandbox,
                            runs_dir=tmp_path / "runs", get_commit=lambda: "abc123",
                            commit_result=lambda run_dir, index_path, status: "not-a-real-commit")

    assert record["status"] == "completed"
    assert record["verify"] == {"ok": True, "problem_count": 0, "problems": []}
    assert record["analysis"] == {"estimate": 1555, "passed": True}

    tool_result = json.loads(client.calls[1]["messages"][2]["content"][0]["content"])
    assert tool_result["exit_code"] == 0 and tool_result["stdout"] == "1555\n"

    outputs = tmp_path / "runs" / record["run_id"] / "outputs" / "solver-0"
    assert (outputs / "solver-0-0001.out").read_text() == "1555\n"
    with tarfile.open(outputs / "solver-0-workspace.tar") as tar:  # read, never extracted
        assert tar.extractfile("./answer.txt").read() == b"1555"

    log = [json.loads(line) for line in (tmp_path / "runs" / record["run_id"] / "log.jsonl").read_text().splitlines()]
    metadata = log[0]["data"]["metadata"]
    assert metadata["sandbox"]["image_id"].startswith("sha256:")
    assert any(p.startswith("numpy==") for p in metadata["sandbox"]["packages"])
    assert {"kind": "container_stop", "data": {"workspace_archived": True}}.items() <= \
        next(e for e in log if e["kind"] == "container_stop").items()
    assert record["storage_bytes"]["workspace_archives"]["solver-0"] > 0


def test_tampered_copy_of_a_real_run_fails_verify(tmp_path):
    client = FakeClient([
        tool_call("run_python", {"code": "print(1555)\n"}),
        tool_call("submit_result", {"total": 1555, "method_notes": "printed it."}),
    ])
    record = run_experiment(TOY_CONFIG, client, Logger, sandbox_factory=Sandbox,
                            runs_dir=tmp_path / "runs", get_commit=lambda: "abc123",
                            commit_result=lambda run_dir, index_path, status: "not-a-real-commit")
    assert record["verify"]["ok"]

    copy = tmp_path / "copy"
    shutil.copytree(tmp_path / "runs" / record["run_id"], copy)
    (copy / "outputs" / "solver-0" / "solver-0-0001.out").write_text("1554\n")
    ok, problems = verify(copy / "log.jsonl", record["final_log_hash"])
    assert not ok
    assert [(p["kind"], p["file"]) for p in problems] == [("file_altered", "solver-0/solver-0-0001.out")]


def test_api_key_appears_nowhere_in_a_run_folder(tmp_path, monkeypatch):
    # The key lives only in the host environment. Nothing written for a run,
    # including the log, outputs and the workspace tar, may contain it.
    secret = "sk-ant-test-must-not-leak-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    client = FakeClient([
        tool_call("run_python", {"code": (
            "import os, json\n"
            "print(json.dumps(dict(os.environ)))\n"
            "open('/work/env.txt', 'w').write(json.dumps(dict(os.environ)))\n")}),
        tool_call("submit_result", {"total": 1555, "method_notes": "looked around."}),
    ])
    record = run_experiment(TOY_CONFIG, client, Logger, sandbox_factory=Sandbox,
                            runs_dir=tmp_path / "runs", get_commit=lambda: "abc123",
                            commit_result=lambda run_dir, index_path, status: "not-a-real-commit")
    assert record["status"] == "completed"
    files = [p for p in (tmp_path / "runs").rglob("*") if p.is_file()]
    assert any(p.name == "log.jsonl" for p in files) and any(p.suffix == ".tar" for p in files)
    for path in files:
        assert secret.encode() not in path.read_bytes(), path
        assert b"sk-ant" not in path.read_bytes(), path
