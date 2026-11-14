"""Tool definitions and their Pydantic input models.

Every tool an agent may call is declared here. The JSON schema sent to the
model comes from the Pydantic model's ``model_json_schema()``; the same model
validates each incoming call with ``model_validate()`` before anything runs.

``submit_result`` is deliberately absent: its input model is supplied per
experiment (see ``experiments/ising/schemas.py``) and wrapped with
``tool_definition`` at run time.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

RUN_PYTHON_DESCRIPTION = (
    "Run Python code in an isolated sandbox. Returns exit code, stdout, stderr, "
    "duration, and whether the run timed out or output was truncated."
)

# Why the agent is making a run_python call, in its own words. Required, so the
# record states the intent of every execution next to its code, whatever the
# API does with text written between tool calls (on current models, longer
# notes come back inside thinking blocks, not as visible text). Whitespace is
# stripped and an empty purpose is rejected; the upper bound only guards
# against pasting code into it. The code itself is never stripped.
PURPOSE_MAX_CHARS = 1000
Purpose = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=PURPOSE_MAX_CHARS)]


def run_python_model(default_timeout_s: int, max_timeout_s: int) -> type[BaseModel]:
    """The ``run_python`` input model, with its timeouts from the config.

    Built per run so the schema the agent is shown, the validation applied
    and the config agree: ``limits.default_timeout_s`` is used when the agent
    passes no timeout_s, and ``limits.max_timeout_s`` is the most it may ask for.
    """

    class RunPythonInput(BaseModel):
        """Input for ``run_python``."""

        # Declared first, so the schema lists it before the code: the agent
        # states what it is about to do, then writes it.
        purpose: Purpose = Field(
            description="One or two sentences: what this call is for and why you are running it now.")
        code: str = Field(description="Python source to execute in the sandbox.")
        timeout_s: int = Field(
            default=default_timeout_s,
            ge=1,
            le=max_timeout_s,
            description="Wall-clock limit in seconds. The process is killed when it is exceeded.",
        )

    return RunPythonInput


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


SEND_MESSAGE = tool_definition(
    "send_message",
    "Send a message to another agent in this run.",
    SendMessageInput,
)
