"""Prompt placeholders: {name} in a prompt file is filled in from the config.

Only a fixed list of names is allowed; anything else stops the run before it
starts. The filled-in text is what is logged and what the agent is sent.
"""

import pytest
import yaml
from pydantic import ValidationError

from harness.agent import Agent
from harness.config import RunConfig, fill_prompt, prompt_values
from harness.run import check_config_files
from harness.tools import DEFAULT_TIMEOUT_S
from tests.conftest import TOY_CONFIG, FakeClient, entries_of, tool_call

GOOD_SUBMISSION = {"total": 1555, "method_notes": "Summed with Python."}


def config_with_prompt(tmp_path, text: str) -> RunConfig:
    prompt = tmp_path / "solver.txt"
    prompt.write_text(text, encoding="utf-8")
    raw = yaml.safe_load(TOY_CONFIG.read_text())
    raw["roles"]["solver"]["system_prompt"] = str(prompt)  # absolute path wins over the repo root
    return RunConfig.model_validate(raw)


def test_every_allowed_placeholder_is_filled_from_config(toy_config):
    text = " ".join("{" + name + "}" for name in prompt_values(toy_config))
    filled = fill_prompt(text, prompt_values(toy_config))
    limits, sandbox = toy_config.limits, toy_config.sandbox
    assert filled.split() == [
        str(limits.max_turns), str(limits.max_submissions), str(limits.max_timeout_s),
        str(DEFAULT_TIMEOUT_S), "1", str(sandbox.memory_mb),
        str(sandbox.work_mb), str(sandbox.tmp_mb), str(sandbox.shm_mb),
    ]


def test_whole_number_cpus_shown_without_decimal(toy_config):
    assert prompt_values(toy_config)["cpus"] == "1"
    half = toy_config.model_copy(update={"sandbox": toy_config.sandbox.model_copy(update={"cpus": 0.5})})
    assert prompt_values(half)["cpus"] == "0.5"


def test_prompt_without_placeholders_is_unchanged(toy_config):
    text = "No numbers here.\nJust text."
    assert fill_prompt(text, prompt_values(toy_config)) == text


@pytest.mark.parametrize("text", [
    "You have {max_turn} turns.",          # misspelled
    "The analysis is {analysis}.",          # a real config field, but not allowed
    "Use a dict like {'T': 2.0}.",           # braces that aren't a placeholder
    "A stray } brace.",
    "A stray { brace.",
    "Doubled {{max_turns}} braces.",
])
def test_unknown_placeholder_or_stray_brace_rejected(toy_config, text):
    with pytest.raises(ValueError, match="line 1"):
        fill_prompt(text, prompt_values(toy_config))


def test_bad_placeholder_stops_run_before_it_starts(tmp_path):
    config = config_with_prompt(tmp_path, "You have {max_turnz} turns.")
    with pytest.raises(ValueError, match="max_turnz"):
        check_config_files(config)


def test_filled_prompt_is_logged_and_sent(tmp_path, make_harness):
    config = config_with_prompt(tmp_path, "At most {max_turns} turns and {max_timeout_s} s per call.")
    harness, logger = make_harness(config=config)
    expected = f"At most {config.limits.max_turns} turns and {config.limits.max_timeout_s} s per call."

    logged = entries_of(logger, "system_prompt")[0]["data"]["text"]
    assert logged == expected

    client = FakeClient([tool_call("run_python", {"code": "print(1)"}),
                         tool_call("submit_result", GOOD_SUBMISSION)])
    harness.start_agent("solver-0")
    Agent("solver-0", client, harness).run(harness.task_for("solver-0"))
    harness.stop_agent("solver-0")
    assert all(call["system"] == expected for call in client.calls)


def test_max_timeout_below_default_rejected():
    raw = yaml.safe_load(TOY_CONFIG.read_text())
    raw["limits"]["max_timeout_s"] = DEFAULT_TIMEOUT_S - 1
    with pytest.raises(ValidationError, match="default timeout"):
        RunConfig.model_validate(raw)
