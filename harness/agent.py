"""Agent: a system prompt, a message history and a tool-use loop.

An agent is a conversation, not a process. One class serves every role; roles
differ only by system prompt and tool list, both set by config and supplied by
the harness.

Each turn:
  1. stop if the agent has an accepted submission, hit its turn limit, or the
     run's token budget is spent;
  2. call the API (streamed; see below) and log the full response;
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
``limits.api_retries`` times, 2, 4, 8, ... seconds apart; anything else ends
the agent at once (status ``api_error``).

If the quick retries run out on such an error, the API is treated as down
and the agent waits it out: the same request is sent again after 30 s, then
60, 120, ... (at most 5 min apart) until ``limits.api_outage_wait_s`` seconds
of waiting are spent. The agent cannot tell: it sees no clock, nothing runs
in its container meanwhile, and the conversation goes on where it stopped.
``api_outage_start`` and ``api_outage_end`` entries bracket the wait. If the
API is still down at the end, the agent ends with status ``api_outage``, and
an independent batch stops instead of starting its next agent into the same
outage. On 2026-11-20 a connection loss of a few minutes ended six agents of
one batch in 30 s, one after another, because each gave up after 6 s.

Every request is streamed: the response arrives as a flow of small pieces
and ``get_final_message()`` assembles them into the same Message that
``messages.create`` would return, so the log and the history are unchanged.
Streaming changes only how the response travels, not what the model sees or
writes. A non-streamed request sends nothing back until the whole response is
ready, and a network can cut a connection that stays silent that long: on
2026-11-17, at effort max, every non-streamed request still thinking after
60 s was dropped (APIConnectionError), and a streamed one ran for 109 s.

Every request asks for a readable summary of the model's reasoning
(``THINKING``). Without it the API returns thinking blocks with empty text, so
the record would show what the agent did but not why. The setting changes only
what is returned and logged: the model thinks, and is billed, the same either
way. It is a harness property, not an experimental variable, so it is fixed
here rather than set by config. The summaries reach the log inside each
``api_response`` and go back to the model unchanged, like every thinking block.

No automatic model fallback is configured. A fallback would silently switch
the model partway through a run, which would confound the experiment. A
refusal is logged and ends the agent.
"""

import time

import anthropic

MAX_RETRY_WAIT_S = 60
OUTAGE_FIRST_WAIT_S = 30   # first wait once the quick retries are used up
OUTAGE_MAX_WAIT_S = 300    # longest single wait during an outage

# Adaptive thinking (the model decides how much to think), with a readable
# summary returned instead of empty thinking text. See the module docstring.
THINKING = {"type": "adaptive", "display": "summarized"}

STATUS_SUBMITTED = "submitted"
STATUS_TURN_LIMIT = "turn_limit"
STATUS_BUDGET = "budget_exhausted"
STATUS_ENDED_TURN = "ended_turn_without_submission"
STATUS_MAX_TOKENS = "max_tokens_without_tool_call"
STATUS_REFUSAL = "refusal"
STATUS_API_ERROR = "api_error"      # an error that retrying can't fix (e.g. a bad request)
STATUS_API_OUTAGE = "api_outage"    # the API stayed unreachable for limits.api_outage_wait_s


class APIOutage(Exception):
    """The API stayed down for the whole outage wait."""


class Agent:
    """One LLM conversation driven to completion through the harness."""

    def __init__(self, agent_id: str, client, harness, sleep=None) -> None:
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
        self.api_outage_wait_s = config.limits.api_outage_wait_s
        self._sleep = sleep or time.sleep  # replaced in tests so retries don't really wait
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
            except APIOutage:
                status = STATUS_API_OUTAGE
                break
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
        """Call the API, retrying transient failures. Every failure is logged.

        Quick retries first (``api_retries``); then, if the error is still
        one that retrying might fix, the outage wait (module docstring).
        """
        attempt = 0
        outage_waits = 0      # waits taken since the outage began
        outage_waited_s = 0   # seconds spent in those waits
        while True:
            attempt += 1
            try:
                with self._client.messages.stream(**self._request()) as stream:
                    response = stream.get_final_message()
            except anthropic.APIError as error:
                retryable = is_retryable(error)
                in_outage = retryable and attempt > self.api_retries
                if not retryable:
                    wait_s = 0
                elif not in_outage:
                    wait_s = retry_wait_s(error, attempt)
                else:
                    remaining_s = self.api_outage_wait_s - outage_waited_s
                    wait_s = min(outage_wait_s(outage_waits + 1), remaining_s)
                    if outage_waits == 0 and wait_s > 0:
                        self._harness.log(self.agent_id, "api_outage_start", {
                            "after_attempts": attempt, "limit_s": self.api_outage_wait_s})
                will_retry = wait_s > 0
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
                    if outage_waits:  # a non-retryable error means the API answered again
                        self._log_outage_end(recovered=not in_outage, attempts=attempt,
                                             waited_s=outage_waited_s)
                    if in_outage:
                        raise APIOutage(f"API unreachable after {attempt} attempts") from error
                    raise
                self._sleep(wait_s)
                if in_outage:
                    outage_waits += 1
                    outage_waited_s += wait_s
                continue
            if outage_waits:
                self._log_outage_end(recovered=True, attempts=attempt, waited_s=outage_waited_s)
            return response

    def _log_outage_end(self, recovered: bool, attempts: int, waited_s: float) -> None:
        self._harness.log(self.agent_id, "api_outage_end", {
            "recovered": recovered, "attempts": attempts, "waited_s": waited_s})

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
            "thinking": THINKING,
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        return request


def make_client() -> anthropic.Anthropic:
    """The API client for real runs. Reads ANTHROPIC_API_KEY from the environment.

    SDK retries are off: retries happen in Agent, where they are logged.
    """
    return anthropic.Anthropic(max_retries=0)


def outage_wait_s(n: int) -> float:
    """The n-th wait of an outage: 30, 60, 120, 240, then 300 s each."""
    return min(OUTAGE_FIRST_WAIT_S * 2 ** (n - 1), OUTAGE_MAX_WAIT_S)


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
