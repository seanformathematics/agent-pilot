"""Guard against the docs drifting from the code.

README §1 cites tests as evidence that each rule is enforced. If a test is
renamed or deleted, the README would quietly claim protection that no
longer exists. This test catches that.
"""

import re
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
