"""The run report: a read-only HTML view of a run folder (harness/report.py)."""

import os

import pytest

from harness.config import repo_path
from harness.logger import Logger
from harness.report import main, render_event, write_report
from harness.run import run_experiment
from tests.conftest import TOY_CONFIG, FakeClient, FakeSandbox, make_message, tool_call

GOOD = {"total": 1555, "method_notes": "Summed with Python's sum()."}
INDEPENDENT_CONFIG = repo_path("tests/fixtures/toy_sum/config_independent.yaml")


def thinking_then_run(summary, purpose, code="print(1)"):
    return make_message([
        {"type": "thinking", "thinking": summary, "signature": "sig-test"},
        {"type": "tool_use", "id": f"toolu_{abs(hash((summary, purpose))) % 10**9}", "name": "run_python",
         "input": {"purpose": purpose, "code": code}},
    ], "tool_use")


def make_run(tmp_path, responses, config=TOY_CONFIG):
    """A complete run folder with a real hash-chained log, plus its index line."""
    record = run_experiment(
        config, FakeClient(responses), Logger, sandbox_factory=FakeSandbox,
        runs_dir=tmp_path / "runs", get_commit=lambda: "abc123",
        commit_result=lambda run_dir, index_path, status: "not-committed-in-tests",
    )
    return tmp_path / "runs" / record["run_id"]


def report_text(run_dir, tmp_path):
    return write_report(run_dir, tmp_path / "reports").read_text(encoding="utf-8")


@pytest.fixture
def single(tmp_path):
    return make_run(tmp_path, [
        thinking_then_run("Plan: add the numbers with Python.", "Sum the list to get the total."),
        tool_call("submit_result", GOOD),
    ])


def test_report_shows_purpose_reasoning_code_submission_and_analysis(single, tmp_path):
    text = report_text(single, tmp_path)
    assert "Verified:" in text
    assert "Sum the list to get the total." in text           # the purpose
    assert "Plan: add the numbers with Python." in text        # the reasoning summary
    assert "print(1)" in text                                  # the code
    assert "Final accepted submission" in text and "1555" in text
    assert "<h2>Analysis</h2>" in text and "passed" in text   # from the index line


def test_every_link_in_the_report_points_at_a_real_file(single, tmp_path):
    import re
    path = write_report(single, tmp_path / "reports")
    links = re.findall(r'href="([^"]+)"', path.read_text(encoding="utf-8"))
    assert links
    assert [link for link in links if not (path.parent / link).exists()] == []


def test_agent_content_is_escaped_never_rendered_as_html(tmp_path):
    run_dir = make_run(tmp_path, [
        thinking_then_run("<img src=x onerror=alert(1)>", "<script>alert('purpose')</script>",
                          code="print('</pre><script>alert(2)</script>')"),
        tool_call("submit_result", GOOD),
    ])
    text = report_text(run_dir, tmp_path)
    assert "<script" not in text and "<img" not in text
    assert "&lt;script&gt;alert(&#x27;purpose&#x27;)&lt;/script&gt;" in text


def test_a_tampered_log_is_reported_first(single, tmp_path):
    log = single / "log.jsonl"
    text = log.read_text(encoding="utf-8")
    assert "Sum the list to get the total." in text
    log.write_text(text.replace("Sum the list to get the total.", "Something else entirely.", 1), encoding="utf-8")
    report = report_text(single, tmp_path)
    assert "Verification FAILED" in report
    assert report.index("Verification FAILED") < report.index("Timeline")


def test_rejected_call_is_shown_with_the_harness_message(tmp_path):
    run_dir = make_run(tmp_path, [
        tool_call("run_python", {"code": "print(1)"}),  # no purpose: rejected
        tool_call("run_python", {"purpose": "Now with a purpose.", "code": "print(1)"}),
        tool_call("submit_result", GOOD),
    ])
    text = report_text(run_dir, tmp_path)
    assert "Rejected by the harness" in text and "Invalid run_python input" in text


def test_independent_run_compares_agents_and_flags_identical_code(tmp_path):
    one_agent = [tool_call("run_python", {"purpose": "Sum it.", "code": "print(1)"}), tool_call("submit_result", GOOD)]
    run_dir = make_run(tmp_path, one_agent * 3, config=INDEPENDENT_CONFIG)
    text = report_text(run_dir, tmp_path)
    assert "Across agents" in text
    for agent in ("solver-0", "solver-1", "solver-2"):
        assert f'id="agent-{agent}"' in text
    assert "solver-0:solver-0-0001.py, solver-1:solver-1-0001.py, solver-2:solver-2-0001.py" in text


def test_report_never_writes_into_the_runs_folder(single, tmp_path):
    runs = single.parent
    before = {p: p.stat().st_mtime_ns for p in runs.rglob("*")}
    write_report(single, tmp_path / "reports")
    assert {p: p.stat().st_mtime_ns for p in runs.rglob("*")} == before
    for inside in (runs, runs / "reports", single):
        with pytest.raises(ValueError, match="inside the runs folder"):
            write_report(single, inside)


def test_records_without_a_special_layout_are_shown_not_dropped(tmp_path):
    shown = render_event({"kind": "some_future_kind", "ts": "t", "data": {"detail": "kept"}}, tmp_path, tmp_path)
    assert "some_future_kind" in shown and "kept" in shown


def test_command_line_rejects_a_missing_folder(tmp_path, capsys):
    assert main(["harness.report", str(tmp_path / "nope"), "--out", str(tmp_path / "reports")]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_command_line_writes_one_report_per_run(single, tmp_path, capsys):
    assert main(["harness.report", str(single), "--out", str(tmp_path / "reports")]) == 0
    written = capsys.readouterr().out.strip()
    assert os.path.basename(written) == f"{single.name}.html"


def test_report_shows_grading_coverage(single, tmp_path):
    assert "Graded 1 of 1 accepted submissions." in report_text(single, tmp_path)


@pytest.mark.parametrize("coverage, expected", [
    (None, "predates the coverage check"),
    ({"accepted": 3, "graded": 1, "complete": False, "ungraded_agents": ["b", "c"]}, "Not graded: b, c."),
    ({"accepted": 1, "graded": None, "complete": None, "ungraded_agents": None,
      "note": "analysis did not report n_graded"}, "Grading coverage unknown: analysis did not report n_graded"),
])
def test_coverage_shortfalls_and_unknowns_are_shown(coverage, expected):
    from harness.report import render_coverage
    line = {"run_id": "x"} if coverage is None else {"run_id": "x", "analysis_coverage": coverage}
    assert expected in render_coverage(line)
