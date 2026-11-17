"""Harness and Agent behaviour against a fake client and fake sandbox."""

import json
import sys

import anthropic
import httpx2 as httpx
import pytest

from harness.agent import Agent
from tests.conftest import FakeClient, entries_of, sent_to_model, text_reply, tool_call

GOOD_SUBMISSION = {"total": 1555, "method_notes": "Summed with Python's sum()."}


def run_agent(make_harness, responses, config=None):
    harness, logger = make_harness(config=config)
    client = FakeClient(responses)
    agent = Agent("solver-0", client, harness)
    harness.start_agent("solver-0")
    outcome = agent.run(harness.task_for("solver-0"))
    harness.stop_agent("solver-0")
    return outcome, harness, logger, client


# ----- the happy path and the record ------------------------------------

def test_run_then_submit_is_accepted(make_harness):
    outcome, harness, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(sum([1, 2]))"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert outcome == {"agent_id": "solver-0", "status": "submitted", "turns": 2}
    assert harness.accepted_submissions == [
        {"agent_id": "solver-0", "attempt": 1, "submission": GOOD_SUBMISSION}]
    assert entries_of(logger, "flag") == []


def test_every_agent_emission_appears_in_log(make_harness):
    responses = [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}, text="Let me compute."),
        tool_call("submit_result", GOOD_SUBMISSION, text="Submitting."),
    ]
    expected_content = [r.to_dict()["content"] for r in responses]
    _, _, logger, _ = run_agent(make_harness, responses)

    logged = [e["data"]["response"]["content"] for e in entries_of(logger, "api_response")]
    assert logged == expected_content

    tool_ids = [block["id"] for content in expected_content for block in content if block["type"] == "tool_use"]
    assert [e["data"]["tool_use_id"] for e in entries_of(logger, "tool_call")] == tool_ids
    assert [e["data"]["tool_use_id"] for e in entries_of(logger, "tool_result")] == tool_ids


def test_run_starts_with_config_and_system_prompts(make_harness):
    _, harness, logger, _ = run_agent(make_harness, [tool_call("submit_result", GOOD_SUBMISSION)])
    first, second = logger.entries[0], logger.entries[1]
    assert first["kind"] == "run_start"
    assert first["data"]["config"]["model"] == "claude-opus-5-5"
    assert second["kind"] == "system_prompt"
    assert second["data"]["text"] == harness.system_prompts["solver"]


def test_task_is_logged_and_sent_first(make_harness):
    _, harness, logger, client = run_agent(make_harness, [tool_call("submit_result", GOOD_SUBMISSION)])
    task = harness.task_for("solver-0")
    assert entries_of(logger, "task")[0]["data"]["text"] == task
    assert client.calls[0]["messages"][0] == {"role": "user", "content": task}


def test_execution_files_archived_and_hashed(make_harness):
    _, harness, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    files = entries_of(logger, "execution")[0]["data"]["files"]
    assert set(files) == {"solver-0-0001.py", "solver-0-0001.out", "solver-0-0001.err", "solver-0-0001.files.jsonl"}
    for name, digest in files.items():
        import hashlib
        assert hashlib.sha256((harness.run_dir / "outputs" / "solver-0" / name).read_bytes()).hexdigest() == digest


# ----- the conversation is built correctly ------------------------------

def test_thinking_blocks_passed_back_unchanged(make_harness):
    first = tool_call("run_python", {"purpose": "test call", "code": "print(1)"})
    _, _, _, client = run_agent(make_harness, [first, tool_call("submit_result", GOOD_SUBMISSION)])
    assistant_turn = client.calls[1]["messages"][1]
    assert assistant_turn == {"role": "assistant", "content": first.to_dict()["content"]}
    assert assistant_turn["content"][0]["type"] == "thinking"


def test_every_request_asks_for_thinking_summaries(make_harness):
    _, _, _, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert len(client.calls) == 2
    assert all(call["thinking"] == {"type": "adaptive", "display": "summarized"} for call in client.calls)


def test_thinking_summary_is_logged_and_passed_back_unchanged(make_harness):
    from tests.conftest import make_message
    summary = "Plan: time one sweep first, then size the production run."
    first = make_message([
        {"type": "thinking", "thinking": summary, "signature": "sig-summary"},
        {"type": "tool_use", "id": "toolu_s", "name": "run_python", "input": {"purpose": "test call", "code": "print(1)"}},
    ], "tool_use")
    _, _, logger, client = run_agent(make_harness, [first, tool_call("submit_result", GOOD_SUBMISSION)])
    # In the log: the first api_response carries the summary text verbatim.
    responses = entries_of(logger, "api_response")
    assert responses[0]["data"]["response"]["content"][0] == {
        "type": "thinking", "thinking": summary, "signature": "sig-summary"}
    # Back to the model: the next request contains the same block, unchanged.
    assert client.calls[1]["messages"][1]["content"][0]["thinking"] == summary


def test_parallel_tool_calls_answered_in_one_message(make_harness):
    from tests.conftest import make_message
    two_calls = make_message([
        {"type": "tool_use", "id": "toolu_a", "name": "run_python", "input": {"purpose": "test call", "code": "print(1)"}},
        {"type": "tool_use", "id": "toolu_b", "name": "run_python", "input": {"purpose": "test call", "code": "fail"}},
    ], "tool_use")
    _, _, _, client = run_agent(make_harness, [two_calls, tool_call("submit_result", GOOD_SUBMISSION)])
    results = client.calls[1]["messages"][2]
    assert results["role"] == "user"
    assert [(r["tool_use_id"], r["is_error"]) for r in results["content"]] == [("toolu_a", False), ("toolu_b", True)]


def test_run_python_result_has_exact_keys(make_harness):
    _, _, _, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    result = json.loads(client.calls[1]["messages"][2]["content"][0]["content"])
    assert list(result) == ["exit_code", "stdout", "stderr", "duration_s", "timed_out", "truncated"]


def test_effort_and_model_come_from_config(make_harness):
    _, _, _, client = run_agent(make_harness, [tool_call("submit_result", GOOD_SUBMISSION)])
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["max_tokens"] == 16000
    assert [t["name"] for t in call["tools"]] == ["run_python", "submit_result"]


# ----- rejected and malformed calls ------------------------------------

def test_malformed_run_python_rejected_and_logged(make_harness):
    _, _, logger, client = run_agent(make_harness, [
        tool_call("run_python", {"source": "print(1)"}),  # wrong key
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    rejection = entries_of(logger, "rejection")[0]["data"]
    assert "Invalid run_python input" in rejection["message"]
    result = client.calls[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert entries_of(logger, "execution") == []


@pytest.mark.parametrize("bad_input", [
    {"code": "print(1)"},                                   # no purpose
    {"purpose": "   ", "code": "print(1)"},                 # blank purpose
    {"purpose": "x" * 1001, "code": "print(1)"},            # over the limit
])
def test_run_python_without_a_real_purpose_is_rejected_and_logged(make_harness, bad_input):
    _, _, logger, client = run_agent(make_harness, [
        tool_call("run_python", bad_input),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    rejection = entries_of(logger, "rejection")[0]["data"]
    assert "Invalid run_python input" in rejection["message"] and "purpose" in rejection["message"]
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert entries_of(logger, "execution") == []  # nothing ran


def test_purpose_is_recorded_with_its_execution(make_harness):
    _, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "  Time one sweep before sizing the run.  ", "code": "print(1)"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert entries_of(logger, "tool_call")[0]["data"]["input"]["purpose"] == "  Time one sweep before sizing the run.  "
    assert entries_of(logger, "execution")[0]["data"]["purpose"] == "Time one sweep before sizing the run."


def test_unknown_tool_rejected_and_logged(make_harness):
    _, _, logger, client = run_agent(make_harness, [
        tool_call("read_logs", {}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert entries_of(logger, "tool_call")[0]["data"]["name"] == "read_logs"
    assert "Unknown or unavailable tool" in entries_of(logger, "rejection")[0]["data"]["message"]
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True


def test_timeout_above_limit_rejected(make_harness, toy_config):
    limit = toy_config.limits.max_timeout_s
    _, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)", "timeout_s": limit + 1}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    message = entries_of(logger, "rejection")[0]["data"]["message"]
    assert "timeout_s" in message and f"less than or equal to {limit}" in message
    assert entries_of(logger, "execution") == []


def test_run_python_schema_shows_config_timeout_limit(make_harness, toy_config):
    """The default and ceiling the agent is shown are the config's, not numbers in the code."""
    harness, _ = make_harness()
    [run_python] = [t for t in harness.tool_definitions("solver-0") if t["name"] == "run_python"]
    timeout = run_python["input_schema"]["properties"]["timeout_s"]
    assert timeout["maximum"] == toy_config.limits.max_timeout_s
    assert timeout["default"] == toy_config.limits.default_timeout_s


# ----- submissions -------------------------------------------------------

def test_submission_without_code_execution_is_flagged(make_harness):
    _, harness, logger, _ = run_agent(make_harness, [tool_call("submit_result", GOOD_SUBMISSION)])
    flags = entries_of(logger, "flag")
    assert [f["data"]["reason"] for f in flags] == ["submit_without_successful_run"]
    assert harness.accepted_submissions  # flagged, not blocked


def test_submission_after_only_failed_execution_is_flagged(make_harness):
    _, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "fail"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert [f["data"]["reason"] for f in entries_of(logger, "flag")] == ["submit_without_successful_run"]


def test_invalid_submission_rejected_then_resubmitted(make_harness):
    outcome, harness, logger, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", {"total": "lots"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert outcome["status"] == "submitted"
    message = entries_of(logger, "rejection")[0]["data"]["message"]
    assert "Invalid submission" in message and "Submissions remaining: 2" in message
    assert harness.accepted_submissions == [
        {"agent_id": "solver-0", "attempt": 2, "submission": GOOD_SUBMISSION}]


def test_sanity_check_rejects_submission(make_harness):
    _, harness, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", {"total": 1555, "method_notes": "   "}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert "method_notes is empty" in entries_of(logger, "rejection")[0]["data"]["message"]
    assert len(harness.accepted_submissions) == 1


def test_submission_limit_enforced(make_harness):
    bad = {"total": "x"}
    outcome, harness, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", bad),
        tool_call("submit_result", bad),
        tool_call("submit_result", bad),
        tool_call("submit_result", GOOD_SUBMISSION),
        text_reply("I give up."),
    ])
    assert harness.accepted_submissions == []
    assert "No submissions left" in entries_of(logger, "rejection")[-1]["data"]["message"]
    assert outcome["status"] == "ended_turn_without_submission"


# ----- stopping ----------------------------------------------------------

def test_stops_when_model_ends_turn_without_tool(make_harness):
    outcome, _, _, _ = run_agent(make_harness, [text_reply("Done, I think.")])
    assert outcome["status"] == "ended_turn_without_submission"


def test_stops_at_max_tokens_without_tool(make_harness):
    outcome, _, _, _ = run_agent(make_harness, [text_reply("Long...", stop_reason="max_tokens")])
    assert outcome["status"] == "max_tokens_without_tool_call"


def test_stops_on_refusal(make_harness):
    outcome, _, logger, _ = run_agent(make_harness, [text_reply("", stop_reason="refusal")])
    assert outcome["status"] == "refusal"
    assert entries_of(logger, "agent_end")[0]["data"]["status"] == "refusal"


def test_turn_limit(make_harness, toy_config):
    config = toy_config.model_copy(update={"limits": toy_config.limits.model_copy(update={"max_turns": 2})})
    outcome, _, _, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("run_python", {"purpose": "test call", "code": "print(2)"}),
    ], config=config)
    assert outcome == {"agent_id": "solver-0", "status": "turn_limit", "turns": 2}
    assert len(client.calls) == 2


def test_token_budget(make_harness, toy_config):
    config = toy_config.model_copy(update={"limits": toy_config.limits.model_copy(update={"max_total_tokens": 200})})
    outcome, harness, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),   # 150 tokens
        tool_call("run_python", {"purpose": "test call", "code": "print(2)"}),   # 300 tokens: over budget
        tool_call("run_python", {"purpose": "test call", "code": "print(3)"}),   # never requested
    ], config=config)
    assert outcome["status"] == "budget_exhausted"
    assert harness.tokens_used == 300
    assert [e["data"]["tokens_used"] for e in entries_of(logger, "budget")] == [150, 300]


def api_status_error(cls, status: int, headers: dict | None = None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls(f"HTTP {status}", response=response, body=None)


def run_with_sleep_log(make_harness, responses, config=None):
    harness, logger = make_harness(config=config)
    waits = []
    agent = Agent("solver-0", FakeClient(responses), harness, sleep=waits.append)
    harness.start_agent("solver-0")
    outcome = agent.run(harness.task_for("solver-0"))
    return outcome, logger, waits


def test_transient_error_retried_and_logged(make_harness):
    outcome, logger, waits = run_with_sleep_log(make_harness, [
        api_status_error(anthropic.InternalServerError, 529),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert outcome["status"] == "submitted"
    assert outcome["turns"] == 1  # a retried call is one turn
    error = entries_of(logger, "api_error")[0]["data"]
    assert (error["attempt"], error["status_code"], error["will_retry"], error["wait_s"]) == (1, 529, True, 2)
    assert waits == [2]


def test_retries_exhausted_ends_agent(make_harness):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    errors = [anthropic.APIConnectionError(request=request) for _ in range(3)]
    outcome, logger, waits = run_with_sleep_log(make_harness, errors)
    assert outcome["status"] == "api_error"
    logged = [e["data"] for e in entries_of(logger, "api_error")]
    assert [e["attempt"] for e in logged] == [1, 2, 3]
    assert [e["will_retry"] for e in logged] == [True, True, False]
    assert waits == [2, 4]


def test_retry_after_header_respected(make_harness):
    _, logger, waits = run_with_sleep_log(make_harness, [
        api_status_error(anthropic.RateLimitError, 429, {"retry-after": "7"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert waits == [7.0]


def test_bad_request_not_retried(make_harness):
    outcome, logger, waits = run_with_sleep_log(make_harness, [
        api_status_error(anthropic.BadRequestError, 400),
    ])
    assert outcome["status"] == "api_error"
    assert entries_of(logger, "api_error")[0]["data"]["retryable"] is False
    assert waits == []


def test_client_with_sdk_retries_refused(make_harness):
    harness, _ = make_harness()
    client = FakeClient([])
    client.max_retries = 2
    with pytest.raises(ValueError, match="max_retries=0"):
        Agent("solver-0", client, harness)


def test_make_client_has_sdk_retries_off(monkeypatch):
    from harness.agent import make_client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert make_client().max_retries == 0


def test_every_request_is_streamed(make_harness):
    # Non-streamed requests that think for over a minute were cut by the
    # network (2026-11-17); the agent must only ever stream.
    _, _, _, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert client.streamed == len(client.calls) == 2


SSE_REPLY = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "Nothing to do."}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 5}}),
    ("message_stop", {"type": "message_stop"}),
]


def test_streamed_reply_from_the_real_sdk_is_logged_whole(make_harness):
    # The real SDK client, fed a canned server-sent-event stream: no network.
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in SSE_REPLY)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = anthropic.Anthropic(api_key="test-key-not-real", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(serve)))
    harness, logger = make_harness()
    agent = Agent("solver-0", client, harness)
    harness.start_agent("solver-0")
    outcome = agent.run(harness.task_for("solver-0"))

    assert sent[0]["stream"] is True
    assert outcome["status"] == "ended_turn_without_submission"
    logged = entries_of(logger, "api_response")[0]["data"]
    response = logged.get("response", logged)
    assert [(block["type"], block["text"]) for block in response["content"]] == [("text", "Nothing to do.")]
    assert response["stop_reason"] == "end_turn"
    assert response["usage"]["output_tokens"] == 5


# ----- isolation of the record from the agent --------------------------

def test_no_log_content_reaches_the_model(make_harness):
    _, harness, logger, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("read_logs", {}),
        tool_call("submit_result", {"total": "bad"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    everything_sent = sent_to_model(client)
    assert str(harness.run_dir) not in everything_sent
    assert "log.jsonl" not in everything_sent
    for entry in logger.entries:
        assert entry["hash"] not in everything_sent
    assert "prev_hash" not in everything_sent


def test_analysis_not_imported_during_run(make_harness):
    sys.modules.pop("tests.fixtures.toy_sum.analysis", None)
    run_agent(make_harness, [tool_call("submit_result", GOOD_SUBMISSION)])
    assert "tests.fixtures.toy_sum.analysis" not in sys.modules




# ----- end of agent: archive, then container, then workspace -------------

def test_workspace_archived_after_agent_ends_then_container_removed(make_harness):
    _, harness, logger, _ = run_agent(make_harness, [tool_call("submit_result", GOOD_SUBMISSION)])
    kinds = [e["kind"] for e in logger.entries]
    order = [kinds.index(k) for k in ("agent_end", "workspace_archive", "container_stop")]
    assert order == sorted(order)
    assert entries_of(logger, "container_stop")[0]["data"] == {"workspace_archived": True}

    archive = entries_of(logger, "workspace_archive")[0]["data"]
    path = harness.run_dir / "outputs" / "solver-0" / archive["file"]
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == archive["sha256"]
    assert archive["contents"]["members"] == 1


# ----- non-finite numbers in submissions -----------------------------------

from tests.conftest import make_message


def config_with_submission_model(toy_config, ref):
    return toy_config.model_copy(update={"submission_model": ref, "sanity_checks": None})


@pytest.mark.parametrize("bad", ["NaN", "nan", "Infinity", "-inf"])
def test_non_finite_submission_rejected_and_logged(make_harness, toy_config, bad):
    config = config_with_submission_model(toy_config, "tests.fixtures.float_schemas:ValueSubmission")
    outcome, harness, logger, client = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", {"value": bad}),
        tool_call("submit_result", {"value": 2.5}),
    ], config=config)
    rejection = entries_of(logger, "rejection")[0]["data"]["message"]
    assert "NaN or infinite: value" in rejection
    assert "Submissions remaining: 2" in rejection
    assert entries_of(logger, "tool_call")[1]["data"]["input"] == {"value": bad}  # raw input kept
    assert harness.accepted_submissions == [
        {"agent_id": "solver-0", "attempt": 2, "submission": {"value": 2.5, "samples": []}}]
    assert outcome["status"] == "submitted"


def test_non_finite_inside_a_list_is_named(make_harness, toy_config):
    config = config_with_submission_model(toy_config, "tests.fixtures.float_schemas:ValueSubmission")
    _, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", {"value": 1.0, "samples": [1.0, "inf"]}),
        text_reply("ok"),
    ], config=config)
    assert "samples[1]" in entries_of(logger, "rejection")[0]["data"]["message"]


def test_divergence_reported_through_schema_flag(make_harness, toy_config):
    config = config_with_submission_model(toy_config, "tests.fixtures.float_schemas:DivergenceAwareSubmission")
    _, harness, _, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", {"is_finite": False}),
    ], config=config)
    assert harness.accepted_submissions[0]["submission"] == {"is_finite": False, "value": None}


def test_finite_float_type_rejects_nan_at_validation(make_harness, toy_config):
    config = config_with_submission_model(toy_config, "tests.fixtures.float_schemas:DivergenceAwareSubmission")
    _, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", {"is_finite": True, "value": "NaN"}),
        text_reply("ok"),
    ], config=config)
    assert "finite number" in entries_of(logger, "rejection")[0]["data"]["message"]


def test_submission_counts_only_once_logged(make_harness):
    harness, logger = make_harness()

    def refuse_submissions(agent, kind, data, original=logger.append):
        if kind == "submission_accepted":
            raise ValueError("logger refused the entry")
        return original(agent, kind, data)

    logger.append = refuse_submissions
    harness.start_agent("solver-0")
    with pytest.raises(ValueError):
        harness.handle_tool_call("solver-0", "toolu_x", "submit_result", GOOD_SUBMISSION)
    assert harness.accepted_submissions == []
    assert not harness.is_done("solver-0")


# ----- host sleep during a call ----------------------------------------------

def test_host_sleep_during_a_call_is_flagged_and_the_agent_carries_on(make_harness):
    outcome, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "slept"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert outcome["status"] == "submitted"
    execution = entries_of(logger, "execution")[0]["data"]
    assert execution["wall_duration_s"] == 60.01 and execution["cleanup"] == "ok"
    flags = [e["data"] for e in entries_of(logger, "flag")]
    assert {"reason": "host_clock_gap", "exec_id": execution["exec_id"],
            "duration_s": 0.01, "wall_duration_s": 60.01} in flags


def test_no_clock_gap_flag_on_an_ordinary_call(make_harness):
    _, _, logger, _ = run_agent(make_harness, [
        tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
        tool_call("submit_result", GOOD_SUBMISSION),
    ])
    assert all(e["data"]["reason"] != "host_clock_gap" for e in entries_of(logger, "flag"))
