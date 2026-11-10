"""Config validation catches mistakes before a run starts."""

import pytest
import yaml
from pydantic import ValidationError

from harness.config import RunConfig, load_object
from tests.conftest import TOY_CONFIG


def toy_raw() -> dict:
    return yaml.safe_load(TOY_CONFIG.read_text())


def test_toy_config_is_valid():
    RunConfig.model_validate(toy_raw())


def test_misspelled_key_rejected():
    raw = toy_raw()
    raw["limits"]["max_turn"] = raw["limits"].pop("max_turns")
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_unknown_role_rejected():
    raw = toy_raw()
    raw["agents"][0]["role"] = "supervisor"
    with pytest.raises(ValidationError, match="unknown role"):
        RunConfig.model_validate(raw)


def test_duplicate_agent_ids_rejected():
    raw = toy_raw()
    raw["agents"].append(dict(raw["agents"][0]))
    with pytest.raises(ValidationError, match="unique"):
        RunConfig.model_validate(raw)


def test_send_message_only_in_hierarchical():
    raw = toy_raw()
    raw["roles"]["solver"]["tools"].append("send_message")
    with pytest.raises(ValidationError, match="hierarchical"):
        RunConfig.model_validate(raw)


def test_unknown_tool_rejected():
    raw = toy_raw()
    raw["roles"]["solver"]["tools"].append("read_file")
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_load_object():
    from tests.fixtures.toy_sum.schemas import SumSubmission
    assert load_object("tests.fixtures.toy_sum.schemas:SumSubmission") is SumSubmission
    with pytest.raises(ValueError):
        load_object("tests.fixtures.toy_sum.schemas.SumSubmission")


def test_tmp_larger_than_memory_rejected():
    raw = toy_raw()
    raw["sandbox"]["tmp_mb"] = raw["sandbox"]["memory_mb"] + 1
    with pytest.raises(ValidationError, match="larger than memory_mb"):
        RunConfig.model_validate(raw)


def test_old_memory_string_rejected():
    raw = toy_raw()
    raw["sandbox"]["memory"] = "1g"
    with pytest.raises(ValidationError):
        RunConfig.model_validate(raw)


def test_work_larger_than_memory_rejected():
    raw = toy_raw()
    raw["sandbox"]["work_mb"] = raw["sandbox"]["memory_mb"] + 1
    with pytest.raises(ValidationError, match="work_mb"):
        RunConfig.model_validate(raw)
