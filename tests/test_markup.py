"""Tests for harness/markup.py, the leaked tool-call markup detector.

The leaking inputs are shortened copies of real logged tool calls from the
Haiku Ising batches (LESSONS.md §9).
"""

import json

from pydantic import BaseModel, Field

from harness.markup import DETECTOR_VERSION, check_tool_input, schema_field_names


class Point(BaseModel):
    T: float
    U: float | None
    reason: str = ""


class Submission(BaseModel):
    points: list[Point]
    error_method: str
    seeds: list[int]
    report: str
    notes: str = ""


SUBMIT_FIELDS = schema_field_names(Submission)
RUN_PYTHON_FIELDS = {"purpose", "code", "timeout_s"}


def test_schema_field_names_include_nested_models():
    assert SUBMIT_FIELDS == {"points", "error_method", "seeds", "report", "notes", "T", "U", "reason"}


def test_clean_submission_is_not_flagged():
    raw = {"points": [{"T": 2.5, "U": 0.38}], "error_method": "Jackknife over 20 blocks; <m^4> and <m^2> estimated jointly.",
           "seeds": [1, 2], "report": "Wolff updates.", "notes": ""}
    assert check_tool_input(raw, SUBMIT_FIELDS) is None


def test_wrong_closing_tag_swallowing_a_required_field_is_confirmed():
    # 2026-11-16_ising_haiku_at_high_independent_001, ising-03, first submit_result
    raw = {"points": [{"T": 2.5, "U": 0.38}], "report": "...",
           "error_method": 'far below the 0.005 target.</error_method>\n<parameter name="seeds">[1507, 1607, 1707]'}
    flag = check_tool_input(raw, SUBMIT_FIELDS)
    assert flag["version"] == DETECTOR_VERSION
    assert flag["tier"] == "confirmed"
    assert flag["swallowed"] == ["seeds"]
    assert flag["fields"] == {"error_method": ["</error_method>", '<parameter name="seeds']}


def test_optional_field_swallowed_into_another_is_confirmed():
    # 2026-11-16_ising_haiku_at_medium_independent_001, ising-06: accepted, notes empty
    raw = {"points": [], "error_method": "x", "seeds": [1],
           "report": 'not bias results at the stated precision.</report>\n<parameter name="notes">The target U_err <= 0.005'}
    flag = check_tool_input(raw, SUBMIT_FIELDS)
    assert flag["tier"] == "confirmed"
    assert flag["swallowed"] == ["notes"]


def test_stray_closing_tag_without_swallowing_is_markup_only():
    # 2026-11-17_ising_haiku_at_high_pinned_independent_001, ising-07
    raw = {"points": [], "error_method": "x", "seeds": [1], "report": "r",
           "notes": "mperature (1000 replicas x 2000 sweeps).</notes>\n"}
    flag = check_tool_input(raw, SUBMIT_FIELDS)
    assert flag["tier"] == "markup_only"
    assert flag["swallowed"] == []
    assert flag["fields"] == {"notes": ["</notes>"]}


def test_code_swallowed_into_purpose_is_confirmed():
    # 2026-11-16_ising_haiku_at_medium_independent_001, ising-05: rejected, no code field
    raw = {"purpose": 'Time a vectorized Metropolis sweep.</purpose>\n<parameter name="code">\nimport numpy as np',
           "timeout_s": 120}
    flag = check_tool_input(raw, RUN_PYTHON_FIELDS)
    assert flag["tier"] == "confirmed"
    assert flag["swallowed"] == ["code"]


def test_optional_timeout_swallowed_into_purpose_is_confirmed():
    # 2026-11-18_ising_haiku_at_xhigh_pinned_independent_001, ising-02: accepted, ran at the 60 s default
    raw = {"purpose": 'rough observables at a few temperatures.</purpose>\n<parameter name="timeout_s">300',
           "code": "print(1)"}
    flag = check_tool_input(raw, RUN_PYTHON_FIELDS)
    assert flag["tier"] == "confirmed"
    assert flag["swallowed"] == ["timeout_s"]


def test_nested_field_closing_tag_inside_a_list_is_found():
    raw = {"points": [{"T": 2.5, "U": None, "reason": "diverged</reason>"}],
           "error_method": "x", "seeds": [1], "report": "r"}
    flag = check_tool_input(raw, SUBMIT_FIELDS)
    assert flag["tier"] == "markup_only"
    assert flag["fields"] == {"points": ["</reason>"]}


def test_code_with_ordinary_tags_is_not_flagged():
    raw = {"purpose": "Write an HTML table.",
           "code": 'print("<table><tr><td>U</td></tr></table></purpose></code>")'}
    assert check_tool_input(raw, RUN_PYTHON_FIELDS) is None


def test_parameter_markup_inside_code_is_flagged():
    raw = {"purpose": "Run it.", "code": 'print(1)\n</parameter>\n<parameter name="timeout_s">300'}
    flag = check_tool_input(raw, RUN_PYTHON_FIELDS)
    assert flag["tier"] == "confirmed"
    assert flag["swallowed"] == ["timeout_s"]
    assert set(flag["fields"]) == {"code"}


def test_field_naming_itself_is_not_counted_as_swallowed():
    raw = {"notes": '<parameter name="notes">duplicated wrapper'}
    flag = check_tool_input(raw, SUBMIT_FIELDS)
    assert flag["tier"] == "markup_only"


def test_unrelated_closing_tags_in_text_are_not_flagged():
    raw = {"points": [], "error_method": "Used </b> and </div> in an HTML note.", "seeds": [1], "report": "r"}
    assert check_tool_input(raw, SUBMIT_FIELDS) is None


def test_non_dict_input_is_checked_as_text():
    flag = check_tool_input('</parameter>', RUN_PYTHON_FIELDS)
    assert flag["fields"] == {"_input": ["</parameter>"]}


def test_no_field_names_still_checks_parameter_tags():
    assert check_tool_input({"a": "fine"}, set()) is None
    assert check_tool_input({"a": "x</parameter>"}, set())["tier"] == "markup_only"


# ----- classify_tool_input: every issue, as separate labels ------------------

from harness.markup import ISSUE_LABELS, classify_tool_input, rows_for_log  # noqa: E402
from harness.tools import run_python_model  # noqa: E402

RunPython = run_python_model(60, 600)


class StrictSubmission(Submission):
    model_config = {"extra": "forbid"}


def test_well_formed_input_has_no_issues():
    result = classify_tool_input({"purpose": "p", "code": "print(1)"}, RunPython)
    assert result["issues"] == [] and result["markup"] is None


def test_required_field_swallowed_by_markup_gets_both_labels():
    raw = {"purpose": 'p.</purpose>\n<parameter name="code">\nimport numpy', "timeout_s": 120}
    result = classify_tool_input(raw, RunPython)
    assert result["issues"] == ["markup", "markup_confirmed", "missing_swallowed"]
    assert result["missing"] == ["code"]


def test_required_field_dropped_without_trace():
    # e.g. 2026-11-17_ising_haiku_at_medium_pinned_independent_002, ising-09
    raw = {"purpose": "Check numba availability.", "timeout_s": 120}
    result = classify_tool_input(raw, RunPython)
    assert result["issues"] == ["missing_untraced"]
    assert result["missing"] == ["code"]


def test_optional_field_swallowed_is_markup_without_a_missing_label():
    raw = {"purpose": 'p.</purpose>\n<parameter name="timeout_s">300', "code": "x"}
    assert classify_tool_input(raw, RunPython)["issues"] == ["markup", "markup_confirmed"]


def test_invented_field_is_unexpected():
    # e.g. 2026-11-16_ising_haiku_at_high_independent_001, ising-02: points_check
    raw = {"points": [], "error_method": "x", "seeds": [1], "report": "r", "points_check": "placeholder"}
    result = classify_tool_input(raw, StrictSubmission)
    assert result["issues"] == ["unexpected_field"]
    assert result["unexpected"] == ["points_check"]


def test_unexpected_field_is_reported_even_where_the_model_would_ignore_it():
    # run_python's model does not forbid extra fields, so validation would accept this silently.
    raw = {"purpose": "p", "code": "x", "timeout": 300}
    assert classify_tool_input(raw, RunPython)["issues"] == ["unexpected_field"]


def test_empty_input_is_empty_and_missing():
    result = classify_tool_input({}, StrictSubmission)
    assert result["issues"] == ["empty_input", "missing_untraced"]
    assert result["missing"] == ["error_method", "points", "report", "seeds"]


def test_missing_fields_can_be_partly_swallowed_and_partly_untraced():
    raw = {"points": [], "report": "r", "error_method": 'x</error_method>\n<parameter name="seeds">[1]'}
    result = classify_tool_input(raw, Submission)
    assert result["issues"] == ["markup", "markup_confirmed", "missing_swallowed"]
    raw.pop("report")
    assert "missing_untraced" in classify_tool_input(raw, Submission)["issues"]


def test_non_object_input():
    result = classify_tool_input("just text", RunPython)
    assert "not_object" in result["issues"] and "missing_untraced" in result["issues"]


def test_every_returned_label_is_declared():
    inputs = [{}, "x", {"purpose": 'p</purpose><parameter name="code">', "zzz": 1}]
    for raw in inputs:
        assert set(classify_tool_input(raw, RunPython)["issues"]) <= set(ISSUE_LABELS)


def test_rows_for_log_reads_a_logged_run(tmp_path):
    log = tmp_path / "log.jsonl"
    entries = [
        {"seq": 0, "ts": "t0", "agent": "a1", "kind": "run_start",
         "data": {"config": {"submission_model": "tests.test_markup:StrictSubmission",
                             "limits": {"default_timeout_s": 60, "max_timeout_s": 600}}}},
        {"seq": 1, "ts": "t1", "agent": "a1", "kind": "tool_call",
         "data": {"tool_use_id": "u1", "name": "run_python", "input": {"purpose": "p", "timeout_s": 120}}},
        {"seq": 2, "ts": "t2", "agent": "a1", "kind": "rejection", "data": {"tool_use_id": "u1", "message": "m"}},
        {"seq": 3, "ts": "t3", "agent": "a1", "kind": "tool_call",
         "data": {"tool_use_id": "u2", "name": "submit_result",
                  "input": {"points": [], "error_method": "x", "seeds": [1], "report": "r"}}},
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    rows = rows_for_log(log, "run-x")
    assert [(r["tool"], r["rejected"], r["issues"]) for r in rows] == [
        ("run_python", True, ["missing_untraced"]),
        ("submit_result", False, []),
    ]
