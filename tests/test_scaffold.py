"""Smoke tests: the layout imports and tool schemas are well formed."""

from harness import tools
from harness.topologies import TOPOLOGIES


def test_modules_import():
    import harness.agent  # noqa: F401
    import harness.harness  # noqa: F401
    import harness.sandbox  # noqa: F401


def test_run_python_schema_has_required_fields():
    schema = tools.run_python_model(max_timeout_s=120).model_json_schema()
    assert "code" in schema["properties"]
    assert schema["properties"]["timeout_s"]["default"] == tools.DEFAULT_TIMEOUT_S


def test_run_python_input_rejects_bad_timeout():
    import pytest
    from pydantic import ValidationError

    model = tools.run_python_model(max_timeout_s=120)
    with pytest.raises(ValidationError):
        model.model_validate({"code": "print(1)", "timeout_s": 0})


def test_all_topologies_registered():
    assert set(TOPOLOGIES) == {"single"}
