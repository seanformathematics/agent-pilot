"""Shared test fixtures.

Unit tests never call the real API. ``FakeClient`` replays scripted responses,
built as real ``anthropic.types.Message`` objects so the agent loop sees
exactly what the SDK would give it. ``FakeSandbox`` stands in for Docker.
"""

import copy
import json
import uuid
from pathlib import Path

import pytest
from anthropic.types import Message

from harness.config import load_config, repo_path
from harness.harness import Harness
from tests.fixtures.dummy_logger import DummyLogger

TOY_CONFIG = repo_path("tests/fixtures/toy_sum/config.yaml")


# ----- scripted API responses ------------------------------------------

def make_message(content: list[dict], stop_reason: str, input_tokens=100, output_tokens=50) -> Message:
    return Message.model_validate({
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })


def tool_call(name: str, tool_input: dict, text: str | None = None, thinking: bool = True) -> Message:
    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": "", "signature": f"sig-{uuid.uuid4().hex[:8]}"})
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}", "name": name, "input": tool_input})
    return make_message(content, "tool_use")


def text_reply(text: str, stop_reason: str = "end_turn") -> Message:
    return make_message([{"type": "text", "text": text}], stop_reason)


class FakeClient:
    """Minimal stand-in for ``anthropic.Anthropic`` with scripted replies."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = self  # so ``client.messages.create`` resolves here

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))  # the agent keeps mutating its history
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


# ----- sandbox without Docker ------------------------------------------

class FakeSandbox:
    """Same interface as harness.sandbox.Sandbox; never runs anything.

    Code that is exactly "fail" exits 1; anything else exits 0 printing "ok".
    """

    def __init__(self, image, work_dir, memory, cpus, pids_limit, output_cap):
        self.work_dir = Path(work_dir)
        self.container_id = None

    def start(self):
        self.work_dir.mkdir(parents=True, exist_ok=False)
        self.container_id = f"fake-{uuid.uuid4().hex[:8]}"

    def stop(self):
        self.container_id = None

    def run_python(self, code, timeout_s, exec_id, output_dir):
        failed = code == "fail"
        stdout, stderr = ("", "boom\n") if failed else ("ok\n", "")
        paths = {}
        for suffix, text in ((".py", code), (".out", stdout), (".err", stderr)):
            path = Path(output_dir) / f"{exec_id}{suffix}"
            path.write_text(text)
            paths[suffix] = str(path)
        result = {"exit_code": 1 if failed else 0, "stdout": stdout, "stderr": stderr,
                  "duration_s": 0.01, "timed_out": False, "truncated": False}
        details = {"end_reason": "exited", "script_path": paths[".py"], "stdout_path": paths[".out"],
                   "stderr_path": paths[".err"], "stdout_bytes": len(stdout), "stderr_bytes": len(stderr)}
        return result, details


# ----- fixtures ---------------------------------------------------------

@pytest.fixture
def toy_config():
    return load_config(TOY_CONFIG)


@pytest.fixture
def make_harness(tmp_path, toy_config):
    """Build a Harness on the toy experiment with a dummy logger.

    Pass ``config`` to override, ``sandbox_factory`` to use real Docker.
    """
    def make(config=None, sandbox_factory=FakeSandbox):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        logger = DummyLogger(run_dir / "log.jsonl")
        harness = Harness(config or toy_config, run_dir, tmp_path / "work", logger, sandbox_factory)
        harness.start_run({"commit": "test", "image_id": "test"})
        return harness, logger

    return make


def entries_of(logger, kind: str) -> list[dict]:
    return [e for e in logger.entries if e["kind"] == kind]


def sent_to_model(client: FakeClient) -> str:
    """Everything the harness ever sent to the model, as one string."""
    return json.dumps(client.calls)
