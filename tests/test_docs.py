"""Guard against the docs drifting from the code.

README §1 cites tests as evidence that each rule is enforced. If a test is
renamed or deleted, the README would quietly claim protection that no
longer exists. This test catches that.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_test_cited_in_readme_exists():
    readme = (ROOT / "README.md").read_text()
    cited = set(re.findall(r"`(test_[a-z0-9_]+)`", readme))
    defined = set()
    for path in (ROOT / "tests").rglob("test_*.py"):
        defined |= set(re.findall(r"^def (test_[a-z0-9_]+)", path.read_text(), re.MULTILINE))
    missing = sorted(cited - defined)
    assert cited, "expected the README to cite tests"
    assert not missing, f"README cites tests that do not exist: {missing}"


def test_every_file_in_claude_md_layout_exists():
    claude = (ROOT / "CLAUDE.md").read_text()
    layout = claude[claude.index("## Layout"):claude.index("## Invariants")]
    names = re.findall(r"^\s+([a-z_]+\.py)\b", layout, re.MULTILINE)
    assert names
    for name in names:
        assert (ROOT / "harness" / name).exists(), f"CLAUDE.md layout lists harness/{name}, which does not exist"


def git_attr(attr: str, path: str) -> str:
    out = subprocess.run(["git", "check-attr", attr, "--", path], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return out.rsplit(": ", 1)[1].strip()


def test_run_records_are_never_line_ending_converted():
    # Git on Windows can rewrite \n as \r\n on checkout. In runs/ that would
    # change the bytes behind every SHA-256 in the log, so verify would report
    # untouched outputs as altered. .gitattributes marks runs/ as not-text.
    for path in ["runs/2026-01-01_x_single_001/log.jsonl",
                 "runs/2026-01-01_x_single_001/outputs/agent_0/exec_001.out",
                 "runs/index.jsonl"]:
        assert git_attr("text", path) == "unset", path


def test_code_and_prompts_check_out_with_unix_line_endings():
    for path in ["harness/run.py", "tests/fixtures/toy_sum/config.yaml", "README.md"]:
        assert git_attr("eol", path) == "lf", path
