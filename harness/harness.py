"""The Harness: tool dispatch, validation, logging and budgets.

One Harness per log, shared by every agent that log records: one per run
for `single`, one per agent for `independent` (each agent then has its own
log and budget). It is the only object that touches the sandboxes, the log
and the token budget. Agents
reach all of these only through ``handle_tool_call``, and the Agent class
records API traffic through ``log`` and ``record_usage``.

Logging is not optional and not visible: nothing returned to an agent ever
contains log contents, log paths or hashes.
"""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from pydantic import ValidationError

from harness import tools
from harness.config import RunConfig, load_object, repo_path
from harness.sandbox import Sandbox, inspect_archive

NON_FINITE_MESSAGE = (
    "Submission not accepted: these fields are NaN or infinite: {fields}. "
    "Non-finite numbers are not accepted. If the quantity diverges or is "
    "undefined, report that the way the submission schema provides."
)

SUBMIT_RESULT_DESCRIPTION = (
    "Submit your final result. The input is checked and you get either an "
    "acknowledgment or the specific problems to fix."
)


class Harness:
    """Validates, executes and logs every tool call for one run."""

    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        logger,
        sandbox_factory=Sandbox,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir).resolve()
        self._logger = logger  # harness-owned; never handed to agents
        self._sandbox_factory = sandbox_factory

        # Experiment code needed during the run. The analysis entry point is
        # deliberately NOT loaded here: it is only imported after the run.
        self.submission_model = load_object(config.submission_model)
        self.sanity_checks = load_object(config.sanity_checks) if config.sanity_checks else None

        # System prompts, read once so the logged text is exactly the sent text.
        self.system_prompts = {
            name: repo_path(role.system_prompt).read_text(encoding="utf-8")
            for name, role in config.roles.items()
        }
        self._agents = {agent.id: agent for agent in config.agents}

        self._sandboxes: dict[str, Sandbox] = {}
        self._exec_counts: dict[str, int] = {}
        self._successful_runs: dict[str, int] = {}
        self._submit_attempts: dict[str, int] = {}
        self._done: set[str] = set()
        self.accepted_submissions: list[dict] = []
        self.workspace_archive_bytes: dict[str, int] = {}
        self.tokens_used = 0

    # ----- run and agent lifecycle -------------------------------------

    def start_run(self, metadata: dict) -> None:
        """Write the first log entries: config, metadata, every system prompt.

        ``metadata`` comes from the caller: git commit, Docker image ID,
        Python versions and anything else that identifies the setup.
        """
        self.log("harness", "run_start", {
            "config": self.config.model_dump(),
            "metadata": metadata,
        })
        for role, text in self.system_prompts.items():
            self.log("harness", "system_prompt", {"role": role, "text": text})

    def start_agent(self, agent_id: str) -> None:
        """Start this agent's sandbox with a fresh, empty, in-memory /work."""
        if agent_id in self._sandboxes:
            raise RuntimeError(f"agent {agent_id!r} already started")
        limits = self.config.limits
        box = self._sandbox_factory(
            image=self.config.sandbox.image,
            memory_mb=self.config.sandbox.memory_mb,
            work_mb=self.config.sandbox.work_mb,
            tmp_mb=self.config.sandbox.tmp_mb,
            shm_mb=self.config.sandbox.shm_mb,
            cpus=self.config.sandbox.cpus,
            pids_limit=self.config.sandbox.pids_limit,
            output_cap=limits.output_cap_chars,
            stored_output_cap=limits.max_stored_output_mb * 1024 * 1024,
        )
        box.start()
        self._sandboxes[agent_id] = box
        self._exec_counts[agent_id] = 0
        self._successful_runs[agent_id] = 0
        self._submit_attempts[agent_id] = 0
        self.log(agent_id, "agent_start", {"container_id": box.container_id})

    def stop_agent(self, agent_id: str) -> None:
        """Finish with an agent's sandbox, after the agent has ended.

        In order: archive /work into the run's outputs, inspect the archive
        without extracting it, then remove the container, which discards
        /work. If archiving fails, that is logged; /work is still discarded.
        """
        box = self._sandboxes.get(agent_id)
        if box is None or box.container_id is None:
            return

        output_dir = self.run_dir / "outputs" / agent_id
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = output_dir / f"{agent_id}-workspace.tar"
        max_bytes = self.config.sandbox.workspace_archive_max_mb * 1024 * 1024

        archived = False
        try:
            info = box.archive_workspace(archive_path, max_bytes)
            archived = True
        except (OSError, subprocess.SubprocessError) as error:
            self.log(agent_id, "workspace_archive_failed", {"error": f"{type(error).__name__}: {error}"})

        if archived:
            self.workspace_archive_bytes[agent_id] = info["bytes"]
            self.log(agent_id, "workspace_archive", {
                "file": archive_path.name,
                "sha256": sha256_file(archive_path),
                "bytes": info["bytes"],
                "max_bytes": max_bytes,
                "truncated": info["truncated"],
                "timed_out": info["timed_out"],
                "tar_exit_code": info["exit_code"],
                "tar_errors": info["tar_errors"],
                "errors_file_sha256": sha256_file(Path(info["errors_path"])),
                "contents": inspect_archive(archive_path),
            })
            if info["truncated"] or info["timed_out"]:
                self.log(agent_id, "flag", {"reason": "workspace_archive_incomplete",
                                            "truncated": info["truncated"], "timed_out": info["timed_out"]})

        box.stop()
        self.log(agent_id, "container_stop", {"workspace_archived": archived})

    # ----- what agents are given ---------------------------------------

    def system_prompt_for(self, agent_id: str) -> str:
        return self.system_prompts[self._agents[agent_id].role]

    def task_for(self, agent_id: str) -> str:
        return self._agents[agent_id].task

    def tool_definitions(self, agent_id: str) -> list[dict]:
        """The tools this agent's role is allowed, in the order the config lists them."""
        definitions = []
        for name in self._allowed_tools(agent_id):
            if name == "run_python":
                definitions.append(tools.RUN_PYTHON)
            elif name == "submit_result":
                definitions.append(tools.tool_definition(
                    "submit_result", SUBMIT_RESULT_DESCRIPTION, self.submission_model))
            elif name == "send_message":
                definitions.append(tools.SEND_MESSAGE)
        return definitions

    # ----- logging and budget ------------------------------------------

    def log(self, agent_id: str, kind: str, data: dict) -> None:
        """Append one entry to the run log."""
        self._logger.append(agent_id, kind, data)

    def record_usage(self, agent_id: str, usage: dict) -> None:
        """Add one API response's token usage to the run total and log it."""
        spent = sum(
            usage.get(key) or 0
            for key in ("input_tokens", "output_tokens",
                        "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        self.tokens_used += spent
        self.log(agent_id, "budget", {
            "tokens_this_call": spent,
            "tokens_used": self.tokens_used,
            "max_total_tokens": self.config.limits.max_total_tokens,
        })

    def budget_exhausted(self) -> bool:
        return self.tokens_used >= self.config.limits.max_total_tokens

    def is_done(self, agent_id: str) -> bool:
        """True once this agent has an accepted submission."""
        return agent_id in self._done

    # ----- tool calls --------------------------------------------------

    def handle_tool_call(self, agent_id: str, tool_use_id: str, name: str, raw_input) -> tuple[str, bool]:
        """Validate and execute one tool call; return ``(content, is_error)``.

        Every call is logged first, whatever happens next. Bad calls are
        returned to the agent as errors; they never raise.
        """
        self.log(agent_id, "tool_call", {"tool_use_id": tool_use_id, "name": name, "input": raw_input})

        if name not in self._allowed_tools(agent_id):
            content, is_error = self._reject(agent_id, tool_use_id, f"Unknown or unavailable tool: {name!r}.")
        elif name == "run_python":
            content, is_error = self._run_python(agent_id, tool_use_id, raw_input)
        elif name == "submit_result":
            content, is_error = self._submit_result(agent_id, tool_use_id, raw_input)
        else:
            content, is_error = self._reject(agent_id, tool_use_id, f"Tool {name!r} is not implemented yet.")

        self.log(agent_id, "tool_result", {"tool_use_id": tool_use_id, "content": content, "is_error": is_error})
        return content, is_error

    def _run_python(self, agent_id: str, tool_use_id: str, raw_input) -> tuple[str, bool]:
        try:
            call = tools.RunPythonInput.model_validate(raw_input)
        except ValidationError as error:
            return self._reject(agent_id, tool_use_id, f"Invalid run_python input:\n{error}")
        if call.timeout_s > self.config.limits.max_timeout_s:
            return self._reject(
                agent_id, tool_use_id,
                f"timeout_s={call.timeout_s} exceeds the limit of {self.config.limits.max_timeout_s} seconds.")

        self._exec_counts[agent_id] += 1
        exec_id = f"{agent_id}-{self._exec_counts[agent_id]:04d}"
        output_dir = self.run_dir / "outputs" / agent_id  # harness-owned, never mounted
        output_dir.mkdir(parents=True, exist_ok=True)
        result, details = self._sandboxes[agent_id].run_python(call.code, call.timeout_s, exec_id, output_dir)

        self.log(agent_id, "execution", {
            "tool_use_id": tool_use_id,
            "exec_id": exec_id,
            "timeout_s": call.timeout_s,
            "result": result,
            "end_reason": details["end_reason"],
            "stdout_bytes": details["stdout_bytes"],
            "stderr_bytes": details["stderr_bytes"],
            "stdout_stored_bytes": details["stdout_stored_bytes"],
            "stderr_stored_bytes": details["stderr_stored_bytes"],
            "filesystem": details["filesystem"],  # summary; full listing is the .files.jsonl
            "files": hash_execution_files(details),
        })
        if details["storage_truncated"]:
            self.log(agent_id, "flag", {"reason": "stored_output_truncated", "exec_id": exec_id})

        succeeded = result["exit_code"] == 0 and not result["timed_out"]
        if succeeded:
            self._successful_runs[agent_id] += 1
        return json.dumps(result), not succeeded

    def _submit_result(self, agent_id: str, tool_use_id: str, raw_input) -> tuple[str, bool]:
        self._submit_attempts[agent_id] += 1
        attempt = self._submit_attempts[agent_id]
        limit = self.config.limits.max_submissions

        if self._successful_runs[agent_id] == 0:
            self.log(agent_id, "flag", {"reason": "submit_without_successful_run",
                                        "tool_use_id": tool_use_id, "attempt": attempt})
        if self.is_done(agent_id):
            return self._reject(agent_id, tool_use_id, "A submission was already accepted.")
        if attempt > limit:
            return self._reject(agent_id, tool_use_id, f"No submissions left (limit {limit}).")

        try:
            submission = self.submission_model.model_validate(raw_input)
        except ValidationError as error:
            return self._reject_submission(agent_id, tool_use_id, attempt, f"Invalid submission:\n{error}")

        # NaN and infinity can't be written to a valid JSON log, and Pydantic
        # turns the strings "NaN" or "inf" into those values without complaint.
        bad_fields = non_finite_fields(submission.model_dump())
        if bad_fields:
            text = NON_FINITE_MESSAGE.format(fields=", ".join(bad_fields))
            return self._reject_submission(agent_id, tool_use_id, attempt, text)

        problems = self.sanity_checks(submission, self.task_for(agent_id)) if self.sanity_checks else []
        if problems:
            text = "Submission not accepted:\n" + "\n".join(f"- {p}" for p in problems)
            return self._reject_submission(agent_id, tool_use_id, attempt, text)

        record = {"agent_id": agent_id, "attempt": attempt, "submission": submission.model_dump(mode="json")}
        # Log first: a submission counts as accepted only once it is in the log.
        self.log(agent_id, "submission_accepted", record)
        self.accepted_submissions.append(record)
        self._done.add(agent_id)
        return "Submission accepted.", False

    def _reject_submission(self, agent_id, tool_use_id, attempt, text) -> tuple[str, bool]:
        remaining = self.config.limits.max_submissions - attempt
        return self._reject(agent_id, tool_use_id, f"{text}\n\nSubmissions remaining: {remaining}.")

    def _reject(self, agent_id: str, tool_use_id: str, message: str) -> tuple[str, bool]:
        self.log(agent_id, "rejection", {"tool_use_id": tool_use_id, "message": message})
        return message, True

    # ----- helpers -----------------------------------------------------

    def _allowed_tools(self, agent_id: str) -> list[str]:
        return self.config.roles[self._agents[agent_id].role].tools


def non_finite_fields(value, path: str = "") -> list[str]:
    """Names of every NaN or infinite float inside nested dicts and lists."""
    if isinstance(value, float) and not math.isfinite(value):
        return [path or "(value)"]
    if isinstance(value, dict):
        return [name for key, item in value.items()
                for name in non_finite_fields(item, f"{path}.{key}" if path else str(key))]
    if isinstance(value, (list, tuple)):
        return [name for index, item in enumerate(value)
                for name in non_finite_fields(item, f"{path}[{index}]")]
    return []


def hash_execution_files(details: dict) -> dict:
    """``{file name: sha256}`` for the .py/.out/.err/.files.jsonl of one execution."""
    return {
        Path(details[key]).name: sha256_file(Path(details[key]))
        for key in ("script_path", "stdout_path", "stderr_path", "listing_path")
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
