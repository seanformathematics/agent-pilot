"""Agent: a system prompt, a message history and a tool-use loop.

An agent is a conversation, not a process. One class serves every role; roles
differ only by system prompt and tool list, both set by config and supplied by
the harness.

Each turn:
  1. stop if the agent has an accepted submission, hit its turn limit, or the
     run's token budget is spent;
  2. call ``client.messages.create`` and log the full response;
  3. append the response content to the history exactly as returned (thinking
     blocks must go back unchanged);
  4. run every requested tool through the harness and send all the
     ``tool_result`` blocks back in one user message.

The agent also ends when the model ends its turn without calling a tool,
when it refuses, or when the API keeps failing. It is never nudged to keep
going: how it ended is recorded, not steered.

API retries belong to the harness, not the SDK, so every failed attempt and
every wait is logged. The client must be made with ``max_retries=0`` (use
``make_client``). Errors that might succeed on a second try (network
problems, rate limits, overload, server errors) are retried up to
``limits.api_retries`` times; anything else ends the agent at once.

No automatic model fallback is configured. A fallback would silently switch
the model partway through a run, which would confound the experiment. A
refusal is logged and ends the agent.
"""

import time

import anthropic

MAX_RETRY_WAIT_S = 60

STATUS_SUBMITTED = "submitted"
STATUS_TURN_LIMIT = "turn_limit"
STATUS_BUDGET = "budget_exhausted"
STATUS_ENDED_TURN = "ended_turn_without_submission"
STATUS_MAX_TOKENS = "max_tokens_without_tool_call"
STATUS_REFUSAL = "refusal"
STATUS_API_ERROR = "api_error"


class Agent:
    """One LLM conversation driven to completion through the harness."""

    def __init__(self, agent_id: str, client, harness, sleep=time.sleep) -> None:
        if getattr(client, "max_retries", 0) != 0:
            raise ValueError("the client must have max_retries=0; the harness does its own logged retries")
        self.agent_id = agent_id
        self._client = client
        self._harness = harness
        config = harness.config
        self.model = config.model
        self.effort = config.effort
        self.max_turns = config.limits.max_turns
        self.max_tokens = config.limits.max_tokens_per_response
        self.api_retries = config.limits.api_retries
        self._sleep = sleep  # replaced in tests so retries don't really wait
        self.system_prompt = harness.system_prompt_for(agent_id)
        self.tools = harness.tool_definitions(agent_id)
        self.messages: list[dict] = []
        self.turns = 0

    def run(self, task: str) -> dict:
        """Run the loop to completion and return ``{"agent_id", "status", "turns"}``."""
        self._harness.log(self.agent_id, "task", {"text": task})
        self.messages.append({"role": "user", "content": task})

        while True:
            status = self._reason_to_stop()
            if status:
                break

            try:
                response = self._create_with_retries()
            except anthropic.APIError:
                status = STATUS_API_ERROR
                break
            self.turns += 1

            response_dict = response.to_dict()
            self._harness.log(self.agent_id, "api_response", {
                "turn": self.turns,
                "request_id": getattr(response, "_request_id", None),
                "response": response_dict,
            })
            self._harness.record_usage(self.agent_id, response_dict.get("usage", {}))
            self.messages.append({"role": "assistant", "content": response_dict["content"]})

            if response.stop_reason == "refusal":
                status = STATUS_REFUSAL
                break

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                status = STATUS_MAX_TOKENS if response.stop_reason == "max_tokens" else STATUS_ENDED_TURN
                break

            results = []
            for block in tool_uses:
                content, is_error = self._harness.handle_tool_call(
                    self.agent_id, block.id, block.name, block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })
            self.messages.append({"role": "user", "content": results})

        outcome = {"agent_id": self.agent_id, "status": status, "turns": self.turns}
        self._harness.log(self.agent_id, "agent_end", outcome)
        return outcome

    def _create_with_retries(self):
        """Call the API, retrying transient failures. Every failure is logged."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._client.messages.create(**self._request())
            except anthropic.APIError as error:
                retryable = is_retryable(error)
                will_retry = retryable and attempt <= self.api_retries
                wait_s = retry_wait_s(error, attempt) if will_retry else 0
                self._harness.log(self.agent_id, "api_error", {
                    "attempt": attempt,
                    "type": type(error).__name__,
                    "status_code": getattr(error, "status_code", None),
                    "message": str(error),
                    "retryable": retryable,
                    "will_retry": will_retry,
                    "wait_s": wait_s,
                })
                if not will_retry:
                    raise
                self._sleep(wait_s)

    def _reason_to_stop(self) -> str | None:
        if self._harness.is_done(self.agent_id):
            return STATUS_SUBMITTED
        if self.turns >= self.max_turns:
            return STATUS_TURN_LIMIT
        if self._harness.budget_exhausted():
            return STATUS_BUDGET
        return None

    def _request(self) -> dict:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "tools": self.tools,
            "messages": self.messages,
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        return request


def make_client() -> anthropic.Anthropic:
    """The API client for real runs. Reads ANTHROPIC_API_KEY from the environment.

    SDK retries are off: retries happen in Agent, where they are logged.
    """
    return anthropic.Anthropic(max_retries=0)


def is_retryable(error: anthropic.APIError) -> bool:
    """Network problems, timeouts, rate limits, overload and server errors."""
    if isinstance(error, anthropic.APIConnectionError):  # includes timeouts
        return True
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    return False


def retry_wait_s(error: anthropic.APIError, attempt: int) -> float:
    """Use the server's retry-after header if given, else 2, 4, 8, ... seconds."""
    if isinstance(error, anthropic.APIStatusError):
        header = error.response.headers.get("retry-after")
        try:
            return min(float(header), MAX_RETRY_WAIT_S)
        except (TypeError, ValueError):
            pass
    return min(2 ** attempt, MAX_RETRY_WAIT_S)
