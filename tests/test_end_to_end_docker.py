"""A full single-agent run of the toy experiment: fake model, real container."""

import json
import subprocess

import pytest

from harness.agent import Agent
from harness.sandbox import Sandbox
from tests.conftest import FakeClient, entries_of, tool_call

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


def test_toy_run_end_to_end(make_harness):
    harness, logger = make_harness(sandbox_factory=Sandbox)
    client = FakeClient([
        tool_call("run_python", {"code": "print(sum([17, -4, 256, 1031, 9, -88, 42, 7, 300, -15]))"}),
        tool_call("submit_result", {"total": 1555, "method_notes": "sum() in Python."}),
    ])
    agent = Agent("solver-0", client, harness)
    harness.start_agent("solver-0")
    try:
        outcome = agent.run(harness.task_for("solver-0"))
    finally:
        harness.stop_agent("solver-0")

    assert outcome["status"] == "submitted"
    result = json.loads(client.calls[1]["messages"][2]["content"][0]["content"])
    assert result["exit_code"] == 0 and result["stdout"] == "1555\n"

    execution = entries_of(logger, "execution")[0]["data"]
    assert execution["end_reason"] == "exited"
    assert (harness.run_dir / "outputs" / "solver-0" / "solver-0-0001.out").read_text() == "1555\n"
    assert entries_of(logger, "flag") == []
