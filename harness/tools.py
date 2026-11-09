"""Tool definitions and their Pydantic input models.

Every tool an agent may call is declared here. The JSON schema sent to the
model comes from the Pydantic model's ``model_json_schema()``; the same model
validates each incoming call with ``model_validate()`` before anything runs.

``submit_result`` is deliberately absent: its input model is supplied per
experiment (see ``experiments/ising/schemas.py``) and wrapped with
``tool_definition`` at run time.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RunPythonInput(BaseModel):
    """Input for ``run_python``."""

    code: str = Field(description="Python source to execute in the sandbox.")
    timeout_s: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Wall-clock limit in seconds. The process is killed when it is exceeded.",
    )


class SendMessageInput(BaseModel):
    """Input for ``send_message`` (hierarchical topology only)."""

    to: str = Field(description="Agent id of the recipient.")
    type: Literal["task", "result", "query", "directive"]
    content: str


# Keys that every run_python result carries, in this order, always present.
RUN_PYTHON_RESULT_KEYS = (
    "exit_code",
    "stdout",
    "stderr",
    "duration_s",
    "timed_out",
    "truncated",
)


def tool_definition(name: str, description: str, model: type[BaseModel]) -> dict:
    """Build the tool dict the Anthropic API expects from a Pydantic model."""
    return {
        "name": name,
        "description": description,
        "input_schema": model.model_json_schema(),
    }


RUN_PYTHON = tool_definition(
    "run_python",
    "Run Python code in an isolated sandbox. Returns exit code, stdout, stderr, "
    "duration, and whether the run timed out or output was truncated.",
    RunPythonInput,
)

SEND_MESSAGE = tool_definition(
    "send_message",
    "Send a message to another agent in this run.",
    SendMessageInput,
)
