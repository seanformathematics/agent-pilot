"""Smoke tests: the layout imports and tool schemas are well formed."""

from harness import tools
from harness.topologies import TOPOLOGIES


def test_modules_import():
    import harness.agent  # noqa: F401
    import harness.harness  # noqa: F401
    import harness.sandbox  # noqa: F401


def test_run_python_schema_has_required_fields():
    schema = tools.RUN_PYTHON["input_schema"]
    assert tools.RUN_PYTHON["name"] == "run_python"
    assert "code" in schema["properties"]
    assert schema["properties"]["timeout_s"]["default"] == 60


def test_run_python_input_rejects_bad_timeout():
    import pytest

    with pytest.raises(Exception):
        tools.RunPythonInput.model_validate({"code": "print(1)", "timeout_s": 0})


def test_all_topologies_registered():
    assert set(TOPOLOGIES) == {"single"}
