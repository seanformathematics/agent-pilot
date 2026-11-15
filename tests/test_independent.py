"""The independent topology: n fresh single-agent runs in one run folder.

Layout ("one harness, one log; folders nest the way harnesses nest"):

    runs/<id>/
      config.yaml  design.md  commit.txt      shared, once
      batch.jsonl  final_hash.txt             top-level hash-chained log
      solver-0/  log.jsonl  final_hash.txt  outputs/solver-0/...
      solver-1/  ...

Fake model and fake sandbox, but the REAL logger and verifier, so the
cross-checks between batch.jsonl and the agents' logs are tested for real.
"""

import json
import shutil

import pytest

from harness.config import repo_path
from harness.logger import Logger
from harness.run import run_experiment, verify_independent
from tests.conftest import FakeClient, FakeSandbox, text_reply, tool_call

CONFIG = repo_path("tests/fixtures/toy_sum/config_independent.yaml")
AGENTS = ["solver-0", "solver-1", "solver-2"]
GOOD = {"total": 1555, "method_notes": "Summed with Python's sum()."}


def solves():
    """One agent's scripted replies: run code, then submit. 300 tokens in all."""
    return [tool_call("run_python", {"purpose": "test call", "code": "print(1)"}), tool_call("submit_result", GOOD)]


def independent_run(tmp_path, responses, sandbox_factory=FakeSandbox):
    return run_experiment(
        CONFIG, FakeClient(responses), Logger,
        sandbox_factory=sandbox_factory,
        runs_dir=tmp_path / "runs",
        get_commit=lambda: "abc123",
        commit_result=lambda run_dir, index_path, status: f"commit-of-{run_dir.name}",
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def finished(tmp_path):
    record = independent_run(tmp_path, solves() * 3)
    return record, tmp_path / "runs" / record["run_id"]


# ----- layout and records ---------------------------------------------------

def test_run_folder_layout(finished):
    record, run_dir = finished
    assert record["run_id"].endswith("_toy_sum_independent_001")
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        ["batch.jsonl", "commit.txt", "config.yaml", "design.md", "final_hash.txt"] + AGENTS)
    assert (run_dir / "config.yaml").read_bytes() == CONFIG.read_bytes()
    for agent in AGENTS:
        sub = run_dir / agent
        assert sorted(p.name for p in sub.iterdir()) == ["final_hash.txt", "log.jsonl", "outputs"]
        assert [p.name for p in (sub / "outputs").iterdir()] == [agent]  # only its own outputs


def test_index_line_covers_the_whole_batch(finished, tmp_path):
    record, run_dir = finished
    [line] = read_jsonl(tmp_path / "runs" / "index.jsonl")
    assert line["topology"] == "independent" and line["n_agents"] == 3
    assert line["status"] == "completed"
    assert line["agent_statuses"] == {a: "submitted" for a in AGENTS}
    assert line["accepted_submissions"] == 3
    assert line["tokens_used"] == 900
    assert line["final_log_hash"] == (run_dir / "final_hash.txt").read_text().strip()
    assert line["verify"] == {"ok": True, "problem_count": 0, "problems": []}
    assert line["analysis"] == {"estimate": 1555, "passed": True, "n_graded": 3, "graded": AGENTS,
                                "completion": {"accepted": 3, "agents": 3}}
    assert line["analysis_coverage"] == {"accepted": 3, "graded": 3, "complete": True, "ungraded_agents": []}
    assert set(line["agent_runs"]) == set(AGENTS)
    for agent, info in line["agent_runs"].items():
        assert info["status"] == "completed" and info["tokens_used"] == 300
        assert info["final_log_hash"] == (run_dir / agent / "final_hash.txt").read_text().strip()
    assert set(line["storage_bytes"]["workspace_archives"]) == set(AGENTS)


def test_batch_log_records_each_agents_final_hash(finished):
    _, run_dir = finished
    entries = read_jsonl(run_dir / "batch.jsonl")
    assert [e["kind"] for e in entries] == ["batch_start"] + ["agent_run_end"] * 3 + ["batch_end"]
    assert entries[0]["data"]["metadata"]["commit"] == "abc123"
    assert entries[0]["data"]["agents"] == AGENTS
    for entry, agent in zip(entries[1:4], AGENTS):
        assert entry["data"]["agent_id"] == agent
        assert entry["data"]["final_log_hash"] == (run_dir / agent / "final_hash.txt").read_text().strip()
    assert entries[-1]["data"]["agent_run_statuses"] == {a: "completed" for a in AGENTS}


def test_each_agent_log_is_a_complete_record_of_that_agent_only(finished):
    _, run_dir = finished
    for agent in AGENTS:
        entries = read_jsonl(run_dir / agent / "log.jsonl")
        assert entries[0]["kind"] == "run_start"
        assert entries[0]["data"]["metadata"]["independent_agent"] == agent
        assert entries[1]["kind"] == "system_prompt"
        assert entries[-1]["kind"] == "run_end"
        assert {e["agent"] for e in entries} == {"harness", agent}


def test_each_agent_has_its_own_fresh_budget(finished):
    _, run_dir = finished
    for agent in AGENTS:
        budgets = [e["data"]["tokens_used"] for e in read_jsonl(run_dir / agent / "log.jsonl")
                   if e["kind"] == "budget"]
        assert budgets == [150, 300]  # counts from zero for every agent


# ----- failures ---------------------------------------------------------------

def test_an_agent_ending_without_submission_does_not_stop_the_others(tmp_path):
    record = independent_run(tmp_path, solves() + [text_reply("I give up.")] + solves())
    assert record["status"] == "completed"
    assert record["agent_statuses"] == {
        "solver-0": "submitted", "solver-1": "ended_turn_without_submission", "solver-2": "submitted"}
    assert record["verify"]["ok"]


def test_analysis_grades_only_accepted_submissions_but_sees_every_agent(tmp_path):
    record = independent_run(tmp_path, solves() + [text_reply("I give up.")] + solves())
    assert record["analysis"]["completion"] == {"accepted": 2, "agents": 3}
    assert record["analysis"]["graded"] == ["solver-0", "solver-2"]
    assert record["analysis"]["passed"] is True       # the roster doesn't change grading
    assert record["analysis_coverage"]["complete"] is True


def test_harness_error_in_one_agent_does_not_stop_the_others(tmp_path):
    class FlakySandbox(FakeSandbox):
        started = 0

        def start(self):
            FlakySandbox.started += 1
            if FlakySandbox.started == 2:
                raise OSError("docker hiccup")
            super().start()

    record = independent_run(tmp_path, solves() * 2, sandbox_factory=FlakySandbox)
    assert record["status"] == "error"
    assert record["agent_runs"]["solver-1"]["status"] == "error"
    assert record["agent_statuses"] == {"solver-0": "submitted", "solver-2": "submitted"}
    assert record["verify"]["ok"]
    run_dir = tmp_path / "runs" / record["run_id"]
    errors = [e for e in read_jsonl(run_dir / "solver-1" / "log.jsonl") if e["kind"] == "run_error"]
    assert "docker hiccup" in errors[0]["data"]["message"]


def test_ctrl_c_stops_the_batch_and_closes_every_log(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        independent_run(tmp_path, solves() + [KeyboardInterrupt()])
    [record] = read_jsonl(tmp_path / "runs" / "index.jsonl")
    run_dir = tmp_path / "runs" / record["run_id"]
    assert record["status"] == "aborted"
    assert record["verify"]["ok"]
    assert sorted(p.name for p in run_dir.iterdir() if p.is_dir()) == ["solver-0", "solver-1"]
    kinds = [(e["kind"], e["data"].get("agent_id")) for e in read_jsonl(run_dir / "batch.jsonl")]
    assert ("agent_not_started", "solver-2") in kinds
    assert read_jsonl(run_dir / "solver-1" / "log.jsonl")[-1]["data"]["status"] == "aborted"
    assert read_jsonl(run_dir / "batch.jsonl")[-1]["data"]["status"] == "aborted"


# ----- tampering with the batch is caught -----------------------------------

def tampered_copy(run_dir, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(run_dir, copy)
    return copy


def edit_once(path, old, new):
    """Change one occurrence of ``old``; fail loudly if it isn't there, so a
    tamper test can never pass because nothing was actually tampered with."""
    text = path.read_text()
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1))


def kinds_and_logs(problems):
    return sorted((p["kind"], p["log"]) for p in problems)


def test_untouched_run_verifies(finished):
    _, run_dir = finished
    assert verify_independent(run_dir, None) == []


def test_deleted_agent_folder_is_caught(finished, tmp_path):
    _, run_dir = finished
    copy = tampered_copy(run_dir, tmp_path)
    shutil.rmtree(copy / "solver-1")
    assert kinds_and_logs(verify_independent(copy, None)) == [("file_missing", "batch.jsonl")]


def test_agent_folder_swapped_for_a_different_valid_run_is_caught(finished, tmp_path):
    # Each subfolder verifies on its own; only batch.jsonl can tell them apart.
    _, run_dir = finished
    copy = tampered_copy(run_dir, tmp_path)
    shutil.rmtree(copy / "solver-1")
    shutil.copytree(run_dir / "solver-2", copy / "solver-1")
    problems = verify_independent(copy, None)
    assert ("final_hash_mismatch", "solver-1/log.jsonl") in kinds_and_logs(problems)


def test_extra_folder_in_the_run_is_caught(finished, tmp_path):
    _, run_dir = finished
    copy = tampered_copy(run_dir, tmp_path)
    shutil.copytree(run_dir / "solver-0", copy / "solver-9")
    problems = verify_independent(copy, None)
    assert [(p["kind"], p["file"]) for p in problems] == [("file_extra", "solver-9")]


def test_edited_agent_log_is_caught_and_attributed(finished, tmp_path):
    _, run_dir = finished
    copy = tampered_copy(run_dir, tmp_path)
    edit_once(copy / "solver-2" / "log.jsonl", '"total":1555', '"total":1554')
    problems = verify_independent(copy, None)
    assert kinds_and_logs(problems) == [("edited_entry", "solver-2/log.jsonl")]


def test_edited_batch_log_is_caught(finished, tmp_path):
    _, run_dir = finished
    copy = tampered_copy(run_dir, tmp_path)
    edit_once(copy / "batch.jsonl", '"status":"completed"', '"status":"error"')
    assert ("edited_entry", "batch.jsonl") in kinds_and_logs(verify_independent(copy, None))


# ----- an analysis that leaves agents ungraded is caught --------------------

def test_agents_left_out_of_the_grading_are_named_in_the_index(tmp_path):
    # LESSONS.md §4: an analysis written for one agent graded only one of many.
    variant = tmp_path / "first_only.yaml"
    text = CONFIG.read_text()
    assert "toy_sum.analysis:analyze\n" in text
    variant.write_text(text.replace("toy_sum.analysis:analyze\n", "toy_sum.analysis:analyze_first_only\n"))
    record = run_experiment(
        variant, FakeClient(solves() * 3), Logger, sandbox_factory=FakeSandbox,
        runs_dir=tmp_path / "runs", get_commit=lambda: "abc123",
        commit_result=lambda run_dir, index_path, status: "not-committed-in-tests",
    )
    assert record["status"] == "completed"  # a shortfall is recorded, not a failed run
    assert record["analysis_coverage"] == {
        "accepted": 3, "graded": 1, "complete": False, "ungraded_agents": ["solver-1", "solver-2"]}
