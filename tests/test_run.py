"""Run setup, the run folder, the index line, and what happens when things fail."""

import json
import subprocess
import sys

import pytest

from harness.run import clean_commit, run_experiment
from tests.conftest import TOY_CONFIG, FakeClient, FakeSandbox, text_reply, tool_call
from tests.fixtures.dummy_logger import DummyLogger

GOOD_SUBMISSION = {"total": 1555, "method_notes": "Summed with Python's sum()."}


def toy_run(tmp_path, responses, config_path=TOY_CONFIG, logger_factory=DummyLogger,
            sandbox_factory=FakeSandbox, client=None):
    return run_experiment(
        config_path,
        client or FakeClient(responses),
        logger_factory,
        sandbox_factory=sandbox_factory,
        runs_dir=tmp_path / "runs",
        get_commit=lambda: "abc123",
    )


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
    record = toy_run(tmp_path, [tool_call("run_python", {"code": "print(1)"}),
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
    record = toy_run(tmp_path, [tool_call("run_python", {"code": "print(1)"}),
                                tool_call("submit_result", GOOD_SUBMISSION)])
    lines = (tmp_path / "runs" / "index.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [record]
    assert record["status"] == "completed"
    assert record["agent_statuses"] == {"solver-0": "submitted"}
    assert record["analysis"] == {"estimate": 1555, "passed": True}
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


# ----- failures still produce a closed log and an index line --------------

def test_failing_analysis_is_recorded(tmp_path):
    variant = config_variant(tmp_path, "toy_sum.analysis:analyze", "toy_sum.analysis:missing")
    record = toy_run(tmp_path, [tool_call("submit_result", GOOD_SUBMISSION)], config_path=variant)
    assert record["status"] == "completed"
    assert record["analysis"] is None
    assert "AttributeError" in record["analysis_error"]


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
    assert record["analysis"] == {"estimate": None, "passed": False}


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
                       runs_dir=tmp_path / "runs", get_commit=dirty)
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
