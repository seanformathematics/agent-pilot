"""Detect tool-call markup that leaked into a tool's input fields.

The model writes each tool call in a markup format that the API turns into
JSON. Sometimes it closes a parameter with the wrong tag, e.g.
``</error_method>`` instead of ``</parameter>``. The parser then reads the next
parameter (its tag and its value) as ordinary text inside the previous field:

    "error_method": "...far below the target.</error_method>\\n<parameter name=\\"seeds\\">[1507, ...]"

Validation can't always see this. If the swallowed field is required, the call
is rejected; if it is optional, the call is accepted with the wrong value
(LESSONS.md §9). This module is the single, versioned rule for spotting it,
used on logged tool inputs. It only reads: it never rejects or changes a call.

The rule, version 1. A call is flagged if:
- any field other than ``code`` contains ``<parameter name=``, ``</parameter>``,
  or a closing tag named after one of the tool's own fields (e.g.
  ``</error_method>``); or
- ``code`` contains ``<parameter name=`` or ``</parameter>``. (Python code may
  legitimately contain other tags, so only these two count there.)

Non-string values (lists, numbers, nested objects) are checked as their JSON
text. A flagged call gets one of two tiers:
- ``confirmed``: a ``<parameter name="X">`` tag naming a *different* field X
  was found inside a field, so X's value ended up in the wrong place.
- ``markup_only``: markup is present, but no field was swallowed (e.g. a
  stray ``</notes>`` at the end of ``notes``).

Markup is one kind of malformed call among several. ``classify_tool_input``
compares an input with the tool's schema as well and returns every issue it
finds as a separate label, so one call can carry several and they can be
grouped and tested for correlation (see ``ISSUE_LABELS``).

If a new kind of leak is found, the rule gets a new version. Results are
always reported with the version that produced them, and earlier versions'
numbers are never replaced.

Command line (read-only; one JSON line per tool call, for analysis):

    python -m harness.markup runs/<run> [runs/<run> ...]
"""

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel

from harness.config import load_object
from harness.tools import run_python_model

DETECTOR_VERSION = 1

PARAMETER_OPEN = re.compile(r'<parameter\s+name\s*=\s*"?([A-Za-z_]\w*)')
PARAMETER_CLOSE = "</parameter>"


def schema_field_names(model: type[BaseModel]) -> set[str]:
    """Every property name in a model's JSON schema, nested models included."""
    names: set[str] = set()

    def collect(schema: dict) -> None:
        names.update(schema.get("properties", {}))
        for definition in schema.get("$defs", {}).values():
            collect(definition)

    collect(model.model_json_schema())
    return names


def check_tool_input(raw_input, field_names) -> dict | None:
    """Return a description of leaked markup in one tool call's input, or None if clean.

    ``raw_input`` is the input exactly as the API delivered it (normally a
    dict). ``field_names`` are the tool's schema property names; they decide
    which bare closing tags count.
    """
    if not isinstance(raw_input, dict):
        raw_input = {"_input": raw_input}
    closing = _closing_tag_pattern(field_names)
    found: dict[str, list[str]] = {}
    swallowed: set[str] = set()
    for field, value in raw_input.items():
        text = value if isinstance(value, str) else json.dumps(value)
        hits = []  # (position, tag), so tags are reported in the order they appear
        for match in PARAMETER_OPEN.finditer(text):
            hits.append((match.start(), match.group(0)))
            if match.group(1) != field:
                swallowed.add(match.group(1))
        hits.extend((match.start(), match.group(0)) for match in re.finditer(re.escape(PARAMETER_CLOSE), text))
        if field != "code" and closing is not None:
            hits.extend((match.start(), match.group(0)) for match in closing.finditer(text))
        if hits:
            found[field] = [tag for _, tag in sorted(hits)]
    if not found:
        return None
    return {
        "version": DETECTOR_VERSION,
        "tier": "confirmed" if swallowed else "markup_only",
        "fields": found,
        "swallowed": sorted(swallowed),
    }


# Every label classify_tool_input can return. They are not exclusive.
ISSUE_LABELS = (
    "not_object",          # the input is not a JSON object at all
    "empty_input",         # the input is {}
    "markup",              # leaked tool-call markup (check_tool_input), either tier
    "markup_confirmed",    # ... and a field was swallowed into another field
    "missing_swallowed",   # a required field is missing, and its tag is inside another field
    "missing_untraced",    # a required field is missing, with no trace of it anywhere
    "unexpected_field",    # a field the schema does not define
)


def classify_tool_input(raw_input, model: type[BaseModel]) -> dict:
    """Every structural issue in one tool call's input, judged against its input model.

    Returns ``issues`` (a sorted list of labels from ``ISSUE_LABELS``, empty
    when the input is well formed), the field names behind each schema issue,
    and the ``markup`` result of ``check_tool_input`` (or None). It looks only
    at the input's structure, not its values: value errors (a negative U, a
    wrong temperature) are for validation and sanity checks. Fields are
    checked at the top level; markup is searched for everywhere.
    """
    schema = model.model_json_schema()
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    markup = check_tool_input(raw_input, schema_field_names(model))
    swallowed = set(markup["swallowed"]) if markup else set()
    issues = set()
    if not isinstance(raw_input, dict):
        issues.add("not_object")
        present = set()
    else:
        present = set(raw_input)
        if not raw_input:
            issues.add("empty_input")
    missing = required - present
    unexpected = present - properties if isinstance(raw_input, dict) else set()
    if markup:
        issues.add("markup")
        if markup["tier"] == "confirmed":
            issues.add("markup_confirmed")
    if missing & swallowed:
        issues.add("missing_swallowed")
    if missing - swallowed:
        issues.add("missing_untraced")
    if unexpected:
        issues.add("unexpected_field")
    return {
        "version": DETECTOR_VERSION,
        "issues": sorted(issues),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
        "markup": markup,
    }


def _closing_tag_pattern(field_names) -> re.Pattern | None:
    names = sorted(set(field_names), key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"</(%s)>" % "|".join(re.escape(name) for name in names))


# ----- command line: one JSON row per logged tool call ----------------------


def rows_for_log(log_path: Path, run_id: str) -> list[dict]:
    """Classify every tool call in one log. Read-only.

    The input models are rebuilt from the config logged in the log's first
    entry, using the schema code in the current checkout (the run's own commit
    is in its ``commit.txt``; check that the schema hasn't changed since).
    """
    entries = [json.loads(line) for line in log_path.read_text().splitlines()]
    config = entries[0]["data"]["config"]
    limits = config["limits"]
    models = {
        "run_python": run_python_model(limits["default_timeout_s"], limits["max_timeout_s"]),
        "submit_result": load_object(config["submission_model"]),
    }
    rejected = {e["data"]["tool_use_id"] for e in entries if e["kind"] == "rejection"}
    rows = []
    for entry in entries:
        if entry["kind"] != "tool_call":
            continue
        call = entry["data"]
        model = models.get(call["name"])
        result = classify_tool_input(call["input"], model) if model else None
        markup = result["markup"] if result else check_tool_input(call["input"], set())
        rows.append({
            "run_id": run_id,
            "agent_id": entry["agent"],
            "seq": entry["seq"],
            "ts": entry["ts"],
            "tool": call["name"],
            "tool_use_id": call["tool_use_id"],
            "rejected": call["tool_use_id"] in rejected,
            "version": DETECTOR_VERSION,
            "issues": result["issues"] if result else (["markup"] if markup else []),
            "missing": result["missing"] if result else [],
            "unexpected": result["unexpected"] if result else [],
            "swallowed": markup["swallowed"] if markup else [],
            "markup_tier": markup["tier"] if markup else None,
        })
    return rows


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for run in argv:
        run_dir = Path(run)
        for log_path in sorted(run_dir.glob("**/log.jsonl")):
            for row in rows_for_log(log_path, run_dir.name):
                print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
