"""Run setup, the run folder, the index line, and what happens when things fail."""

import json
import subprocess
import sys

import pytest

from harness.run import clean_commit, commit_run, run_experiment
from tests.conftest import TOY_CONFIG, FakeClient, FakeSandbox, text_reply, tool_call
from tests.fixtures.dummy_logger import DummyLogger

GOOD_SUBMISSION = {"total": 1555, "method_notes": "Summed with Python's sum()."}


def record_commit(run_dir, index_path, status):
    return f"commit-of-{run_dir.name}"


def clean_verifier(log_path, expected_final_hash):
    return True, []


def toy_run(tmp_path, responses, config_path=TOY_CONFIG, logger_factory=DummyLogger,
            sandbox_factory=FakeSandbox, client=None, commit_result=record_commit,
            verifier=clean_verifier):
    return run_experiment(
        config_path,
        client or FakeClient(responses),
        logger_factory,
        sandbox_factory=sandbox_factory,
        runs_dir=tmp_path / "runs",
        get_commit=lambda: "abc123",
        commit_result=commit_result,
        verifier=verifier,
    )


def index_part(outcome):
    """The returned outcome minus the git fields, which are not in the index."""
    return {k: v for k, v in outcome.items() if k not in ("git_commit", "git_commit_error")}


def read_log(run_dir):
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


def config_variant(tmp_path, old: str, new: str):
    text = TOY_CONFIG.read_text()
    assert old in text
    path = tmp_path / "variant.yaml"
    path.write_text(text.replace(old, new))
    return path


# ----- the run folder and index ------------------------------------------

def test_run_folder_contents(tmp_path):
    record = toy_run(tmp_path, [tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
                                tool_call("submit_result", GOOD_SUBMISSION)])
    run_dir = tmp_path / "runs" / record["run_id"]
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "commit.txt", "config.yaml", "design.md", "final_hash.txt", "log.jsonl", "outputs"]
    assert (run_dir / "config.yaml").read_bytes() == TOY_CONFIG.read_bytes()
    design = TOY_CONFIG.parent / "design.md"
    assert (run_dir / "design.md").read_bytes() == design.read_bytes()
    assert (run_dir / "commit.txt").read_text() == "abc123\n"
    assert (run_dir / "final_hash.txt").read_text() == record["final_log_hash"] + "\n"


def test_index_line(tmp_path):
    record = toy_run(tmp_path, [tool_call("run_python", {"purpose": "test call", "code": "print(1)"}),
                                tool_call("submit_result", GOOD_SUBMISSION)])
    lines = (tmp_path / "runs" / "index.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [index_part(record)]
    assert record["status"] == "completed"
    assert record["agent_statuses"] == {"solver-0": "submitted"}
    assert record["analysis"] == {"estimate": 1555, "passed": True, "n_graded": 1, "graded": ["solver-0"],
                                  "completion": {"accepted": 1, "agents": 1}}
    assert record["analysis_error"] is None
    assert record["commit"] == "abc123"
    assert record["tokens_used"] == 300
    storage = record["storage_bytes"]
    assert storage["log"] > 0 and storage["outputs"] > 0
    assert set(storage["workspace_archives"]) == {"solver-0"}


def test_run_ids_count_up(tmp_path):
    first = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)])
    second = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)])
    assert first["run_id"].endswith("_toy_sum_single_001")
    assert second["run_id"].endswith("_toy_sum_single_002")
    assert len((tmp_path / "runs" / "index.jsonl").read_text().splitlines()) == 2


def test_log_starts_with_setup_then_prompts_and_ends_with_run_end(tmp_path):
    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)])
    entries = read_log(tmp_path / "runs" / record["run_id"])
    assert entries[0]["kind"] == "run_start"
    metadata = entries[0]["data"]["metadata"]
    assert metadata["commit"] == "abc123"
    assert metadata["sandbox"]["image_id"] == "sha256:fake"
    assert "anthropic_sdk" in metadata and "host_python" in metadata
    assert entries[1]["kind"] == "system_prompt"
    assert entries[-1]["kind"] == "run_end"
    assert entries[-1]["data"]["status"] == "completed"


def test_analysis_result_not_in_log(tmp_path):
    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)])
    log_text = (tmp_path / "runs" / record["run_id"] / "log.jsonl").read_text()
    assert '"passed"' not in log_text


def test_analysis_not_imported_until_run_ends(tmp_path):
    sys.modules.pop("tests.fixtures.toy_sum.analysis", None)
    seen = []

    class CheckingClient(FakeClient):
        def create(self, **kwargs):
            seen.append("tests.fixtures.toy_sum.analysis" in sys.modules)
            return super().create(**kwargs)

    record = toy_run(tmp_path, None, client=CheckingClient([tool_call("submit_result", GOOD_SUBMISSION)]))
    assert seen == [False]
    assert record["analysis"]["passed"] is True  # imported afterwards


# ----- did the analysis grade every accepted submission? ------------------

def test_analysis_coverage_is_complete_when_every_submission_is_graded(tmp_path):
    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)])
    assert record["analysis_coverage"] == {"accepted": 1, "graded": 1, "complete": True, "ungraded_agents": []}
    line = json.loads((tmp_path / "runs" / "index.jsonl").read_text().splitlines()[-1])
    assert line["analysis_coverage"] == record["analysis_coverage"]  # it is in the index line


def test_analysis_coverage_is_unknown_when_the_analysis_does_not_report_it(tmp_path):
    variant = config_variant(tmp_path, "toy_sum.analysis:analyze", "toy_sum.analysis:analyze_without_coverage")
    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], config_path=variant)
    assert record["analysis_error"] is None
    assert record["analysis_coverage"]["complete"] is None
    assert record["analysis_coverage"]["note"] == "analysis did not report n_graded"


@pytest.mark.parametrize("result, complete, ungraded", [
    ({"n_graded": 2, "graded": ["a", "b"]}, True, []),
    ({"n_graded": 1, "graded": ["a"]}, False, ["b"]),          # one agent left out
    ({"n_graded": 2, "graded": ["a", "a"]}, False, ["b"]),     # right count, wrong agents
    ({"n_graded": 2}, True, None),                             # count only: trusted, ids unknown
    ({"n_graded": True}, None, None),                          # a bool is not a count
])
def test_analysis_coverage_compares_what_was_graded_with_what_was_accepted(result, complete, ungraded):
    from harness.run import analysis_coverage
    submissions = [{"agent_id": "a"}, {"agent_id": "b"}]
    coverage = analysis_coverage(submissions, {"estimate": None, "passed": True, **result})
    assert coverage["accepted"] == 2
    assert coverage["complete"] is complete
    assert coverage["ungraded_agents"] == ungraded


# ----- the analysis also gets the full list of agents ----------------------

def test_analysis_is_told_about_an_agent_that_did_not_submit(tmp_path):
    record = toy_run(tmp_path, [text_reply("I give up.")])
    assert record["accepted_submissions"] == 0
    # The toy analysis counts agents from the roster; grading is unchanged.
    assert record["analysis"]["completion"] == {"accepted": 0, "agents": 1}
    assert record["analysis"]["passed"] is False
    assert record["analysis"]["n_graded"] == 0


def test_agent_roster_lists_every_configured_agent_in_config_order():
    from types import SimpleNamespace
    from harness.run import agent_roster
    config = SimpleNamespace(agents=[SimpleNamespace(id=a) for a in ["a", "b", "c"]])
    outcomes = [{"agent_id": "a", "status": "submitted"}, {"agent_id": "b", "status": "turn_limit"}]
    submissions = [{"agent_id": "a", "attempt": 1, "submission": {}}]
    assert agent_roster(config, outcomes, submissions) == [
        {"agent_id": "a", "status": "submitted", "accepted": True},
        {"agent_id": "b", "status": "turn_limit", "accepted": False},
        {"agent_id": "c", "status": "not_run", "accepted": False},   # run ended before c reported
    ]


# ----- failures still produce a closed log and an index line --------------

def test_failing_analysis_is_recorded(tmp_path):
    variant = config_variant(tmp_path, "toy_sum.analysis:analyze", "toy_sum.analysis:missing")
    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], config_path=variant)
    assert record["status"] == "completed"
    assert record["analysis"] is None
    assert "AttributeError" in record["analysis_error"]
    assert record["analysis_coverage"]["complete"] is None
    assert record["analysis_coverage"]["note"] == "analysis failed; see analysis_error"


def test_harness_error_recorded_and_tidied(tmp_path):
    class BrokenSandbox(FakeSandbox):
        def start(self):
            raise OSError("docker is not running")

    record = toy_run(tmp_path, [], sandbox_factory=BrokenSandbox)
    assert record["status"] == "error"
    entries = read_log(tmp_path / "runs" / record["run_id"])
    error = next(e for e in entries if e["kind"] == "run_error")
    assert "docker is not running" in error["data"]["message"]
    assert entries[-1]["kind"] == "run_end"
    assert (tmp_path / "runs" / record["run_id"] / "final_hash.txt").exists()


def test_ctrl_c_marks_run_aborted_then_reraises(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        toy_run(tmp_path, [KeyboardInterrupt()])
    record = json.loads((tmp_path / "runs" / "index.jsonl").read_text())
    assert record["status"] == "aborted"
    run_dir = tmp_path / "runs" / record["run_id"]
    assert read_log(run_dir)[-1]["data"]["status"] == "aborted"
    kinds = [e["kind"] for e in read_log(run_dir)]
    assert "container_stop" in kinds  # the container was still stopped


def test_silent_logger_refused_before_run_folder(tmp_path):
    class SilentLogger(DummyLogger):
        def append(self, agent, kind, data):
            return "nothing written"

    with pytest.raises(RuntimeError, match="logger wrote nothing"):
        toy_run(tmp_path, [], logger_factory=SilentLogger)
    assert not (tmp_path / "runs").exists()


def test_agent_outcome_other_than_submitted_still_completes_run(tmp_path):
    record = toy_run(tmp_path, [text_reply("I am done.")])
    assert record["status"] == "completed"
    assert record["agent_statuses"] == {"solver-0": "ended_turn_without_submission"}
    assert record["analysis"] == {"estimate": None, "passed": False, "n_graded": 0, "graded": [],
                                  "completion": {"accepted": 0, "agents": 1}}


# ----- refused before any run folder exists -------------------------------

def test_missing_prompt_file_refused_before_run_folder(tmp_path):
    variant = config_variant(tmp_path, "prompts/solver.txt", "prompts/nope.txt")
    with pytest.raises(FileNotFoundError):
        toy_run(tmp_path, [], config_path=variant)
    assert not (tmp_path / "runs").exists()


def test_unimplemented_topology_refused_before_run_folder(tmp_path):
    variant = config_variant(tmp_path, "topology: single", "topology: one_shot")
    with pytest.raises(NotImplementedError):
        toy_run(tmp_path, [], config_path=variant)
    assert not (tmp_path / "runs").exists()


def test_dirty_tree_refused_before_run_folder(tmp_path):
    def dirty():
        raise RuntimeError("git working tree has uncommitted changes")

    with pytest.raises(RuntimeError):
        run_experiment(TOY_CONFIG, FakeClient([]), DummyLogger, sandbox_factory=FakeSandbox,
                       runs_dir=tmp_path / "runs", get_commit=dirty, commit_result=record_commit,
                       verifier=clean_verifier)
    assert not (tmp_path / "runs").exists()


# ----- the clean-tree check itself, in a scratch git repository ----------

def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def scratch_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "code.py").write_text("x = 1\n")
    (repo / "runs").mkdir()
    (repo / "runs" / ".gitkeep").write_text("")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def test_clean_tree_returns_commit(scratch_repo):
    commit = clean_commit(scratch_repo)
    assert len(commit) == 40


def test_modified_file_refused(scratch_repo):
    (scratch_repo / "code.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        clean_commit(scratch_repo)


def test_untracked_file_refused(scratch_repo):
    (scratch_repo / "new_prompt.txt").write_text("hi")
    with pytest.raises(RuntimeError, match="uncommitted"):
        clean_commit(scratch_repo)


def test_new_runs_do_not_count_as_dirty(scratch_repo):
    (scratch_repo / "runs" / "2026-11-09_toy_single_001").mkdir()
    (scratch_repo / "runs" / "2026-11-09_toy_single_001" / "log.jsonl").write_text("{}\n")
    (scratch_repo / "runs" / "index.jsonl").write_text("{}\n")
    assert len(clean_commit(scratch_repo)) == 40


# ----- committing each run -------------------------------------------------

def test_every_run_is_committed_with_folder_and_index(tmp_path):
    calls = []

    def capture(run_dir, index_path, status):
        calls.append((run_dir.name, index_path.name, status))
        return "deadbeef"

    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], commit_result=capture)
    assert calls == [(record["run_id"], "index.jsonl", "completed")]
    assert record["git_commit"] == "deadbeef" and record["git_commit_error"] is None


def test_failed_and_aborted_runs_are_committed_too(tmp_path):
    calls = []

    def capture(run_dir, index_path, status):
        calls.append(status)
        return "deadbeef"

    class BrokenSandbox(FakeSandbox):
        def start(self):
            raise OSError("docker is not running")

    toy_run(tmp_path, [], sandbox_factory=BrokenSandbox, commit_result=capture)
    with pytest.raises(KeyboardInterrupt):
        toy_run(tmp_path, [KeyboardInterrupt()], commit_result=capture)
    assert calls == ["error", "aborted"]


def test_commit_failure_reported_and_record_untouched(tmp_path):
    def failing(run_dir, index_path, status):
        raise subprocess.CalledProcessError(1, ["git", "commit"], stderr="hook rejected")

    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], commit_result=failing)
    assert record["git_commit"] is None
    assert "hook rejected" in record["git_commit_error"]
    assert record["status"] == "completed"
    assert (tmp_path / "runs" / record["run_id"] / "final_hash.txt").exists()
    assert len((tmp_path / "runs" / "index.jsonl").read_text().splitlines()) == 1


def test_commit_run_commits_only_the_run(scratch_repo):
    run_dir = scratch_repo / "runs" / "2026-11-10_toy_single_001"
    run_dir.mkdir()
    (run_dir / "log.jsonl").write_text("{}\n")
    index = scratch_repo / "runs" / "index.jsonl"
    index.write_text("{}\n")
    (scratch_repo / "code.py").write_text("x = 99\n")      # an unrelated edit...
    git(scratch_repo, "add", "code.py")                     # ...even one already staged

    commit = commit_run(run_dir, index, "completed", repo=scratch_repo)

    shown = subprocess.run(["git", "show", "--name-only", "--format=%s", commit],
                           cwd=scratch_repo, capture_output=True, text=True, check=True).stdout.split()
    assert shown[0] == "run:"
    assert "runs/2026-11-10_toy_single_001/log.jsonl" in shown and "runs/index.jsonl" in shown
    assert "code.py" not in shown
    status = subprocess.run(["git", "status", "--porcelain"], cwd=scratch_repo,
                            capture_output=True, text=True, check=True).stdout
    assert "code.py" in status  # still pending, not swept into the run commit


# ----- the verifier ------------------------------------------------------------

def test_verify_called_with_log_and_final_hash_and_recorded(tmp_path):
    seen = []

    def spy(log_path, expected_final_hash):
        if log_path.name == "log.jsonl":
            seen.append((log_path.parent.name, expected_final_hash))
        return True, []

    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], verifier=spy)
    assert seen == [(record["run_id"], record["final_log_hash"])]
    assert record["verify"] == {"ok": True, "problem_count": 0, "problems": []}


def test_many_problems_counted_first_few_kept(tmp_path):
    problems = [{"kind": "edited_entry", "line": n, "detail": "hash mismatch"} for n in range(1, 151)]

    def damaged(log_path, expected_final_hash):
        if log_path.name == "check.jsonl":
            return True, []  # the trial log is fine
        return False, problems

    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], verifier=damaged)
    assert record["verify"]["ok"] is False
    assert record["verify"]["problem_count"] == 150
    assert record["verify"]["problems"] == problems[:10]
    assert record["status"] == "completed"  # how the run went and whether its record verifies are separate
    assert record["git_commit"]             # a failed verification is still committed, as evidence


def test_crashing_verifier_reported_as_a_problem(tmp_path):
    def crashing(log_path, expected_final_hash):
        if log_path.name == "check.jsonl":
            return True, []
        raise ValueError("boom")

    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], verifier=crashing)
    assert record["verify"]["ok"] is False
    assert record["verify"]["problems"][0]["kind"] == "verifier_error"


def test_unimplemented_verifier_refused_before_run_folder(tmp_path):
    def skeleton(log_path, expected_final_hash):
        return None  # what `...` returns

    with pytest.raises(RuntimeError, match="verifier rejected"):
        toy_run(tmp_path, [], verifier=skeleton)
    assert not (tmp_path / "runs").exists()


def test_verifier_rejecting_clean_trial_log_refused(tmp_path):
    def always_unhappy(log_path, expected_final_hash):
        return False, [{"kind": "edited_entry", "line": 1, "detail": "?"}]

    with pytest.raises(RuntimeError, match="verifier rejected"):
        toy_run(tmp_path, [], verifier=always_unhappy)
    assert not (tmp_path / "runs").exists()


# ----- pre-run API check (command line only) ---------------------------------

import anthropic  # noqa: E402

from harness import run as run_module  # noqa: E402
from tests.test_harness import api_status_error  # noqa: E402

KEY = {"ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key"}


class ModelsClient:
    """Just enough client for check_api_access: models.retrieve()."""

    def __init__(self, error=None):
        self.asked = []
        outer = self

        class Models:
            def retrieve(self, model):
                outer.asked.append(model)
                if error:
                    raise error
                return {"id": model}

        self.models = Models()


def test_api_check_passes_with_key_and_known_model():
    client = ModelsClient()
    assert run_module.check_api_access(client, "claude-opus-5-5", KEY) is None
    assert client.asked == ["claude-opus-5-5"]


def test_api_check_refuses_missing_key_without_calling_the_api():
    client = ModelsClient()
    for environ in ({}, {"ANTHROPIC_API_KEY": "  "}):
        assert "not set" in run_module.check_api_access(client, "m", environ)
    assert client.asked == []


@pytest.mark.parametrize("cls, status, words", [
    (anthropic.AuthenticationError, 401, "rejected"),
    (anthropic.PermissionDeniedError, 403, "may not use"),
    (anthropic.NotFoundError, 404, "not found"),
    (anthropic.InternalServerError, 500, "could not reach"),
])
def test_api_check_explains_failures_without_showing_the_key(cls, status, words):
    message = run_module.check_api_access(ModelsClient(api_status_error(cls, status)), "m", KEY)
    assert words in message
    assert KEY["ANTHROPIC_API_KEY"] not in message


def test_failed_api_check_creates_no_run_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    started = []
    monkeypatch.setattr(run_module, "run_experiment", lambda *a, **k: started.append(a))
    assert run_module.main(["harness.run", str(TOY_CONFIG)]) == 2
    assert started == [] and not (tmp_path / "runs").exists()
    assert "no run folder created" in capsys.readouterr().err


# ---- command line ----------------------------------------------------------

def test_help_prints_usage_and_starts_nothing(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(run_module, "run_experiment", lambda *a, **k: started.append(a))
    with pytest.raises(SystemExit) as exit_info:
        run_module.main(["harness.run", "--help"])
    assert exit_info.value.code == 0
    assert "config" in capsys.readouterr().out
    assert started == []


@pytest.mark.parametrize("argv", [["harness.run"], ["harness.run", "a.yaml", "b.yaml"]])
def test_wrong_number_of_arguments_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_module.main(argv)
    assert exit_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_missing_config_file_gives_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_module, "RUNS_DIR", tmp_path / "runs")
    clients = []
    monkeypatch.setattr("harness.agent.make_client", lambda: clients.append(1))
    assert run_module.main(["harness.run", str(tmp_path / "no_such_config.yaml")]) == 2
    err = capsys.readouterr().err
    assert "no run folder created" in err and "no_such_config.yaml" in err
    assert clients == []  # the API client is never created for a bad config
    assert not (tmp_path / "runs").exists()
