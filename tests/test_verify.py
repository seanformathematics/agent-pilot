"""M2 spec for harness/verify.py (hand-written by Sean), using harness/logger.py.

These tests skip themselves until both files work. They encode the contract
agreed on 2026-11-10:

    verify(log_path, expected_final_hash=None) -> (ok, problems)

``problems`` lists EVERY problem found, not just the first. Each problem is a
dict with at least:
    "kind"   one of the kinds below
    "line"   1-based line number in log.jsonl, or None if not about one line
    "detail" short human-readable explanation
File problems also carry "file" (path relative to the outputs folder).

Checks are local, so one tampered line is reported once, not cascaded:
  * each line's stored hash must match its recomputed hash ("edited_entry")
  * each line's prev_hash must equal the previous line's STORED hash, and
    its seq must be one more ("broken_link", one per location)
A line that isn't valid JSON is "unreadable_line"; checking resumes after
it (the next line's link can't be checked and is not reported). A last line
without a newline is "incomplete_last_line". A final hash that differs from
the expected one is "final_hash_mismatch". Files in the outputs folder next
to the log are checked against the hashes in "execution" and
"workspace_archive" entries: "file_altered", "file_missing", "file_extra".
verify never raises: anything unexpected becomes a reported problem.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

PROBLEM_KINDS = {
    "edited_entry", "broken_link", "unreadable_line", "incomplete_last_line",
    "final_hash_mismatch", "file_altered", "file_missing", "file_extra",
}


def _record_keepers_ready() -> bool:
    try:
        from harness.logger import Logger
        from harness.verify import verify
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "log.jsonl"
            logger = Logger(path)
            logger.append("agent-0", "note", {})
            final = logger.close()
            return path.stat().st_size > 0 and verify(path, final) is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _record_keepers_ready(),
    reason="harness/logger.py and harness/verify.py are not implemented yet (M2)",
)


# ----- helpers -------------------------------------------------------------

def make_log(tmp_path, entries=10):
    from harness.logger import Logger
    path = tmp_path / "log.jsonl"
    logger = Logger(path)
    for i in range(entries):
        logger.append("agent-0", "note", {"i": i})
    return path, logger.close()


def lines_of(path):
    return path.read_text().splitlines(keepends=True)


def write_lines(path, lines):
    path.write_text("".join(lines))


def edit_data(path, line_number, new_data):
    lines = lines_of(path)
    entry = json.loads(lines[line_number - 1])
    entry["data"] = new_data
    lines[line_number - 1] = json.dumps(entry) + "\n"
    write_lines(path, lines)


def check(path, expected_final_hash=None):
    from harness.verify import verify
    ok, problems = verify(path, expected_final_hash)
    assert isinstance(ok, bool) and isinstance(problems, list)
    for problem in problems:
        assert problem["kind"] in PROBLEM_KINDS, problem
        assert "line" in problem and isinstance(problem["detail"], str), problem
    assert ok == (problems == [])
    return {(p["line"], p["kind"]) for p in problems}, problems


# ----- the chain -------------------------------------------------------------

def test_clean_log_verifies(tmp_path):
    path, final = make_log(tmp_path)
    assert check(path, final)[0] == set()


def test_one_edit_reported_once_at_its_line(tmp_path):
    path, final = make_log(tmp_path)
    edit_data(path, 4, {"i": "tampered"})
    assert check(path, final)[0] == {(4, "edited_entry")}


def test_two_independent_edits_both_reported_without_cascade(tmp_path):
    path, final = make_log(tmp_path)
    edit_data(path, 3, {"i": "x"})
    edit_data(path, 7, {"i": "y"})
    assert check(path, final)[0] == {(3, "edited_entry"), (7, "edited_entry")}


def test_line_not_in_canonical_form_is_an_edit(tmp_path):
    # Same parsed content, different bytes: a reader of the raw file could be
    # shown something other than what was hashed (here, a duplicate "kind" key
    # that json.loads silently drops). Only the logger's exact bytes are valid.
    path, final = make_log(tmp_path)
    lines = lines_of(path)
    lines[2] = lines[2].replace('"kind":', '"kind":"innocent","kind":', 1)
    lines[5] = json.dumps(json.loads(lines[5]), sort_keys=True) + "\n"  # adds spaces
    write_lines(path, lines)
    assert check(path, final)[0] == {(3, "edited_entry"), (6, "edited_entry")}


def test_deleted_line_reported_once_where_the_gap_is(tmp_path):
    path, final = make_log(tmp_path)
    lines = lines_of(path)
    del lines[3]  # remove line 4; the old line 5 becomes line 4
    write_lines(path, lines)
    assert check(path, final)[0] == {(4, "broken_link")}


def test_reordered_lines_reported_as_broken_links(tmp_path):
    path, final = make_log(tmp_path)
    lines = lines_of(path)
    lines[3], lines[4] = lines[4], lines[3]
    write_lines(path, lines)
    found, _ = check(path, final)
    assert {kind for _, kind in found} == {"broken_link"}
    assert {4, 5} <= {line for line, _ in found} <= {4, 5, 6}


def test_truncated_log_caught_by_final_hash(tmp_path):
    path, final = make_log(tmp_path)
    write_lines(path, lines_of(path)[:-2])
    assert check(path, final)[0] == {(None, "final_hash_mismatch")}
    assert check(path)[0] == set()  # without the recorded hash, truncation is invisible


def test_wrong_expected_hash_reported(tmp_path):
    path, _ = make_log(tmp_path)
    assert check(path, "0" * 64)[0] == {(None, "final_hash_mismatch")}


def test_half_written_last_line(tmp_path):
    path, _ = make_log(tmp_path)
    text = path.read_text()
    path.write_text(text[: len(text) - 20])  # cut inside the last line, no newline
    found, _ = check(path)
    last = len(lines_of(path))
    assert (last, "incomplete_last_line") in found
    assert {line for line, _ in found} == {last}


def test_unreadable_line_reported_and_checking_resumes(tmp_path):
    path, final = make_log(tmp_path)
    lines = lines_of(path)
    lines[3] = "this is not json\n"
    write_lines(path, lines)
    edit_data(path, 8, {"i": "tampered"})
    assert check(path, final)[0] == {(4, "unreadable_line"), (8, "edited_entry")}


def test_garbage_file_reported_not_raised(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(bytes(range(256)) * 10)
    found, _ = check(path)
    assert found  # some problem reported, and no exception


# ----- output files --------------------------------------------------------

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run_with_outputs(tmp_path):
    from harness.logger import Logger
    outputs = tmp_path / "outputs" / "agent-0"
    outputs.mkdir(parents=True)
    out = outputs / "agent-0-0001.out"
    out.write_text("1555\n")
    tar = outputs / "agent-0-workspace.tar"
    tar.write_bytes(b"pretend tar")
    tar_err = outputs / "agent-0-workspace.tar.err"
    tar_err.write_text("")

    path = tmp_path / "log.jsonl"
    logger = Logger(path)
    logger.append("agent-0", "execution", {"files": {out.name: sha256(out)}})
    logger.append("agent-0", "workspace_archive", {
        "file": tar.name, "sha256": sha256(tar), "errors_file_sha256": sha256(tar_err)})
    return path, logger.close(), outputs


def test_matching_outputs_verify(tmp_path):
    path, final, _ = make_run_with_outputs(tmp_path)
    assert check(path, final)[0] == set()


def test_altered_output_file(tmp_path):
    path, final, outputs = make_run_with_outputs(tmp_path)
    (outputs / "agent-0-0001.out").write_text("1556\n")
    found, problems = check(path, final)
    assert {kind for _, kind in found} == {"file_altered"}
    assert problems[0]["file"] == "agent-0/agent-0-0001.out"


def test_missing_output_file(tmp_path):
    path, final, outputs = make_run_with_outputs(tmp_path)
    (outputs / "agent-0-workspace.tar").unlink()
    assert {kind for _, kind in check(path, final)[0]} == {"file_missing"}


def test_extra_output_file(tmp_path):
    path, final, outputs = make_run_with_outputs(tmp_path)
    (outputs / "planted.txt").write_text("not from the harness")
    found, problems = check(path, final)
    assert {kind for _, kind in found} == {"file_extra"}
    assert problems[0]["file"] == "agent-0/planted.txt"
