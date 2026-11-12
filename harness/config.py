"""Loading and validating experiment configs.

A config is a YAML file (see tests/fixtures/toy_sum/config.yaml for the shape).
Every section is a Pydantic model with ``extra="forbid"``, so a misspelled key
is an error instead of being silently ignored.

Paths inside a config are relative to the repository root. Experiment code is
referenced as "module.path:name" and loaded with ``load_object``.
"""

import importlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parent.parent

ToolName = Literal["run_python", "submit_result", "send_message"]
AGENT_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleConfig(Strict):
    system_prompt: str  # path to a text file; see fill_prompt for {placeholders}
    tools: list[ToolName]


class AgentConfig(Strict):
    id: str = Field(pattern=AGENT_ID_PATTERN)
    role: str
    task: str


class Limits(Strict):
    max_turns: int = Field(ge=1)  # API calls per agent
    max_tokens_per_response: int = Field(ge=1)
    max_total_tokens: int = Field(ge=1)  # input + output, per harness: the whole run, or each agent in independent
    default_timeout_s: int = Field(ge=1)  # run_python timeout when the agent passes none
    max_timeout_s: int = Field(ge=1)  # upper bound on run_python timeout_s
    output_cap_bytes: int = Field(ge=100)  # per stream, per execution, shown to the agent
    max_stored_output_mb: int = Field(ge=1)  # per stream, per execution, saved to disk
    max_submissions: int = Field(ge=1)  # every submit_result call counts
    api_retries: int = Field(ge=0)  # retries of a failed API call; each one is logged

    @model_validator(mode="after")
    def default_timeout_allowed(self):
        # Otherwise every run_python call without its own timeout_s is rejected.
        if self.default_timeout_s > self.max_timeout_s:
            raise ValueError(
                f"default_timeout_s ({self.default_timeout_s}) must not exceed "
                f"max_timeout_s ({self.max_timeout_s})")
        return self


class SandboxConfig(Strict):
    image: str
    memory_mb: int = Field(ge=64)  # everything in the container: processes and the contents of /work, /tmp, /dev/shm
    work_mb: int = Field(ge=1)     # size of /work (in memory, counts against memory_mb)
    tmp_mb: int = Field(ge=1)      # size of /tmp (in memory, counts against memory_mb)
    shm_mb: int = Field(ge=1)      # size of /dev/shm (in memory, counts against memory_mb)
    cpus: float = Field(gt=0)
    pids_limit: int = Field(ge=16)
    workspace_archive_max_mb: int = Field(ge=1)  # cap on the tar of /work kept after the agent ends

    @model_validator(mode="after")
    def folders_fit_in_memory(self):
        for name in ("work_mb", "tmp_mb", "shm_mb"):
            if getattr(self, name) > self.memory_mb:
                raise ValueError(f"{name} ({getattr(self, name)}) is larger than memory_mb ({self.memory_mb})")
        return self


class RunConfig(Strict):
    experiment: str = Field(pattern=AGENT_ID_PATTERN)  # used in run folder names
    design: str
    topology: Literal["single", "independent", "one_shot", "hierarchical"]
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    submission_model: str
    sanity_checks: str | None = None
    analysis: str
    roles: dict[str, RoleConfig]
    agents: list[AgentConfig] = Field(min_length=1)
    limits: Limits
    sandbox: SandboxConfig

    @model_validator(mode="after")
    def check_consistency(self):
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"agent ids must be unique: {ids}")
        for agent in self.agents:
            if agent.role not in self.roles:
                raise ValueError(f"agent {agent.id!r} has unknown role {agent.role!r}")
        if self.topology == "single" and len(self.agents) != 1:
            raise ValueError(f"the single topology needs exactly one agent, got {len(self.agents)}")
        if self.topology == "independent":
            # Independent runs repeat ONE task so the agents' behaviour can be
            # compared; different tasks would confound that.
            if len({(a.role, a.task) for a in self.agents}) != 1:
                raise ValueError("the independent topology needs every agent to have the same role and task")
        for name, role in self.roles.items():
            if "send_message" in role.tools and self.topology != "hierarchical":
                raise ValueError(f"role {name!r}: send_message is only allowed in the hierarchical topology")
        return self


def load_config(path: Path) -> RunConfig:
    """Read and validate a YAML config file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)


def repo_path(relative: str) -> Path:
    """Turn a path written in a config into an absolute path."""
    return REPO_ROOT / relative


def prompt_values(config: RunConfig) -> dict[str, str]:
    """The only values a prompt may contain as {name}, all taken from the config.

    A fixed list on purpose: prompts can state limits the agent works within,
    but never reach other config fields (such as the analysis entry point).
    """
    limits, sandbox = config.limits, config.sandbox
    return {
        "max_turns": str(limits.max_turns),
        "max_submissions": str(limits.max_submissions),
        "max_timeout_s": str(limits.max_timeout_s),
        "default_timeout_s": str(limits.default_timeout_s),
        "cpus": f"{sandbox.cpus:g}",  # 1.0 -> "1", 0.5 -> "0.5"
        "memory_mb": str(sandbox.memory_mb),
        "work_mb": str(sandbox.work_mb),
        "tmp_mb": str(sandbox.tmp_mb),
        "shm_mb": str(sandbox.shm_mb),
    }


def fill_prompt(text: str, values: dict[str, str]) -> str:
    """Replace each {name} in a prompt with its value from ``prompt_values``.

    Strict: once the known names are replaced, any brace left over (an
    unknown or misspelled name, or a stray "{" or "}") is an error, so the
    agent is never sent a literal placeholder. Plain replacement rather than
    str.format, so nothing else in the text is interpreted.
    """
    for name, value in values.items():
        text = text.replace("{" + name + "}", value)
    leftovers = [
        f"line {number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), start=1)
        if "{" in line or "}" in line
    ]
    if leftovers:
        raise ValueError(
            "prompt has unknown placeholders or stray braces "
            f"(allowed: {', '.join('{' + n + '}' for n in values)}):\n" + "\n".join(leftovers))
    return text


def load_prompt(path: str, config: RunConfig) -> str:
    """Read a prompt file and fill in its placeholders from the config."""
    return fill_prompt(repo_path(path).read_text(encoding="utf-8"), prompt_values(config))


def load_object(ref: str):
    """Load a Python object from a "module.path:name" string."""
    module_path, sep, name = ref.partition(":")
    if not sep or not module_path or not name:
        raise ValueError(f"expected 'module.path:name', got {ref!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)
