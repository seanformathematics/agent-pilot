"""Run report: one self-contained HTML page per run folder.

    python -m harness.report runs/<run id> [runs/<run id> ...] [--out reports/]

The report is a *view* of the record, never part of it:
  - it reads the run folder and the run's line in ``runs/index.jsonl`` and
    writes only to the output folder (default ``reports/``, git-ignored);
    it refuses to write anywhere inside the runs folder;
  - it re-runs verification when it is made and puts the result first, so a
    broken hash chain is the first thing the page says;
  - every record in each log appears on the page: records it has no special
    layout for are shown as raw JSON rather than dropped;
  - workspace archives are described from what the log recorded about them,
    never extracted;
  - everything an agent wrote (text, reasoning summaries, purposes, code,
    output) is untrusted and is HTML-escaped. The page loads nothing from
    outside and contains no scripts.

It knows nothing about physics. The analysis section shows whatever the
experiment's analysis entry point put in the index line.
"""

import argparse
import html
import json
import os
import sys
from pathlib import Path

from harness.config import REPO_ROOT
from harness.run import RUNS_DIR, call_verifier, verify_independent
from harness.verify import verify

REPORTS_DIR = REPO_ROOT / "reports"
PREVIEW_CHARS = 6000  # longest stdout/stderr/code shown inline; full files are linked

# Records shown inside the tool-call panel they belong to (matched by tool_use_id).
TOOL_RECORD_KINDS = ("tool_call", "execution", "rejection", "tool_result")


# ----- reading the run folder ------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    """Every parseable line of a JSONL file. Unparseable lines become a record
    of kind ``unreadable_line`` so they still appear on the page."""
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"kind": "unreadable_line", "agent": "", "data": {"line": number, "text": line}})
    return records


def verification_problems(run_dir: Path) -> list[dict]:
    """The same checks the harness runs at the end of a run, redone now."""
    if (run_dir / "batch.jsonl").exists():
        return verify_independent(run_dir, None)
    try:
        final_hash = (run_dir / "final_hash.txt").read_text().strip()
    except OSError as error:
        return [{"kind": "file_missing", "line": None, "detail": f"final_hash.txt: {error}"}]
    return call_verifier(verify, run_dir / "log.jsonl", final_hash)


def find_index_line(run_dir: Path, index_path: Path) -> dict | None:
    """The run's line in the index (the last one, if somehow there are several)."""
    if not index_path.is_file():
        return None
    found = None
    for record in read_jsonl(index_path):
        if record.get("run_id") == run_dir.name:
            found = record
    return found


def agent_folders(run_dir: Path) -> list[tuple[str, Path]]:
    """(label, folder holding log.jsonl and outputs/) for each log in the run."""
    if (run_dir / "batch.jsonl").exists():
        return [(p.name, p) for p in sorted(run_dir.iterdir()) if (p / "log.jsonl").is_file()]
    return [("", run_dir)]


# ----- small HTML helpers ----------------------------------------------------

def e(value) -> str:
    """Escape anything for HTML. Used on every value that came from a record."""
    return html.escape(str(value), quote=True)


def pre(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = text or ""
    note = ""
    if len(text) > limit:
        note = f'<div class="muted">… {len(text) - limit:,} more characters not shown here.</div>'
        text = text[:limit]
    return f"<pre>{e(text)}</pre>{note}"


def details(summary: str, body: str, open_: bool = False) -> str:
    return f'<details{" open" if open_ else ""}><summary>{summary}</summary>{body}</details>'


def link(path: Path, out_dir: Path, label: str) -> str:
    return f'<a href="{e(os.path.relpath(path, out_dir))}">{e(label)}</a>'


def table(rows: list[dict]) -> str:
    """A list of dicts as a table; columns in order of first appearance."""
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{e(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{scalar(row.get(c, ''))}</td>" for c in columns) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def scalar(value) -> str:
    if isinstance(value, float):
        return e(f"{value:.6g}")
    if isinstance(value, (dict, list)):
        return f"<code>{e(json.dumps(value))}</code>"
    return e(value)


def show_value(value) -> str:
    """A generic view of any JSON value: lists of dicts become tables,
    dicts become key/value lists, everything else is printed."""
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        return table(value)
    if isinstance(value, dict):
        items = "".join(f"<tr><th>{e(k)}</th><td>{show_value(v)}</td></tr>" for k, v in value.items())
        return f'<table class="kv">{items}</table>'
    if isinstance(value, str) and ("\n" in value or len(value) > 160):
        return pre(value)
    return scalar(value)


def kv(pairs: list[tuple[str, object]]) -> str:
    rows = "".join(f"<tr><th>{e(k)}</th><td>{scalar(v)}</td></tr>" for k, v in pairs if v is not None)
    return f'<table class="kv">{rows}</table>'


# ----- one agent's timeline --------------------------------------------------

def tool_records(records: list[dict]) -> dict[str, dict[str, dict]]:
    """{tool_use_id: {kind: record}} for the records shown inside tool panels."""
    by_id: dict[str, dict[str, dict]] = {}
    for record in records:
        tool_use_id = record.get("data", {}).get("tool_use_id") if isinstance(record.get("data"), dict) else None
        if record.get("kind") in TOOL_RECORD_KINDS and tool_use_id:
            by_id.setdefault(tool_use_id, {})[record["kind"]] = record
    return by_id


def render_tool_use(block: dict, linked: dict[str, dict], folder: Path, out_dir: Path) -> str:
    name, tool_input = block.get("name", "?"), block.get("input") or {}
    parts = [f'<div class="tool"><div class="toolhead">Tool call: <b>{e(name)}</b></div>']

    if name == "run_python":
        purpose = tool_input.get("purpose")
        parts.append(f'<div class="purpose"><span class="label">Purpose</span> {e(purpose)}</div>' if purpose
                     # Only possible in runs made before run_python required a purpose
                     # (commit feba0c0); later calls without one are rejected.
                     else '<div class="muted">No purpose recorded (this call predates the required purpose field).</div>')
        if "timeout_s" in tool_input:
            parts.append(f'<div class="muted">timeout_s requested: {e(tool_input["timeout_s"])}</div>')
        parts.append(details(f"Code ({len(str(tool_input.get('code', ''))):,} characters)",
                             pre(str(tool_input.get("code", "")))))
    else:
        parts.append(details("Input", show_value(tool_input), open_=(name == "submit_result")))

    if "rejection" in linked:
        parts.append(f'<div class="bad"><b>Rejected by the harness:</b>{pre(linked["rejection"]["data"].get("message", ""))}</div>')

    if "execution" in linked:
        ex = linked["execution"]["data"]
        result = ex.get("result", {})
        state = "bad" if result.get("exit_code") != 0 or result.get("timed_out") else "good"
        parts.append(f'<div class="{state}">exit code {e(result.get("exit_code"))} · '
                     f'{e(result.get("duration_s"))} s · end reason {e(ex.get("end_reason"))}'
                     f'{" · TIMED OUT" if result.get("timed_out") else ""}'
                     f'{" · output truncated for the agent" if result.get("truncated") else ""}</div>')
        exec_id = ex.get("exec_id", "")
        outputs = folder / "outputs" / (linked["execution"].get("agent") or "")
        files = ex.get("files", {})
        if files:
            parts.append('<div class="muted">Stored files (hashed in the log): ' + " · ".join(
                link(outputs / name_, out_dir, name_) for name_ in files) + "</div>")
        for stream in ("stdout", "stderr"):
            text = result.get(stream, "")
            if text:
                parts.append(details(f"{stream} as the agent saw it ({len(text):,} characters; "
                                     f"{e(ex.get(stream + '_bytes'))} bytes produced)", pre(text), open_=(stream == "stdout")))
        fs = ex.get("filesystem", {}).get("dirs", {})
        if fs:
            parts.append('<div class="muted">After the call: ' + "; ".join(
                f'{e(d)} {e(v.get("files"))} files, {e(v.get("file_bytes"))} bytes' for d, v in fs.items())
                + (f' · {link(outputs / (exec_id + ".files.jsonl"), out_dir, "full listing")}' if exec_id else "")
                + "</div>")

    if "tool_result" in linked and "execution" not in linked:
        result = linked["tool_result"]["data"]
        css = "bad" if result.get("is_error") else "good"
        parts.append(f'<div class="{css}"><b>Returned to the agent:</b>{pre(str(result.get("content", "")))}</div>')

    parts.append("</div>")
    return "".join(parts)


def render_turn(record: dict, by_id: dict, folder: Path, out_dir: Path) -> str:
    data = record["data"]
    response = data.get("response", {})
    usage = response.get("usage", {})
    head = (f'Turn {e(data.get("turn"))} · {e(usage.get("input_tokens"))} in / {e(usage.get("output_tokens"))} out'
            f' · stop: {e(response.get("stop_reason"))} · {e(record.get("ts", ""))}')
    parts = [f'<section class="turn"><h4>{head}</h4>']
    for block in response.get("content", []):
        kind = block.get("type")
        if kind == "thinking":
            text = block.get("thinking", "")
            parts.append(f'<div class="thinking"><span class="label">Reasoning summary</span>{pre(text)}</div>' if text
                         else '<div class="muted">Thinking block with no text returned.</div>')
        elif kind == "redacted_thinking":
            parts.append('<div class="muted">Redacted thinking block.</div>')
        elif kind == "text":
            parts.append(f'<div class="said"><span class="label">Visible text</span>{pre(block.get("text", ""))}</div>')
        elif kind == "tool_use":
            parts.append(render_tool_use(block, by_id.get(block.get("id"), {}), folder, out_dir))
        else:
            parts.append(f'<div class="muted">Block of type {e(kind)}:</div>{pre(json.dumps(block, indent=1))}')
    parts.append("</section>")
    return "".join(parts)


def render_event(record: dict, folder: Path, out_dir: Path) -> str:
    """Records that are not turns or tool records, in log order."""
    kind, data = record.get("kind"), record.get("data", {})
    stamp = f'<span class="muted">{e(record.get("ts", ""))}</span>'
    if kind == "system_prompt":
        return details(f'System prompt (role: {e(data.get("role"))})', pre(data.get("text", "")))
    if kind == "task":
        return f'<div class="task"><span class="label">Task</span>{pre(data.get("text", ""))}</div>'
    if kind == "budget":
        return f'<div class="event muted">Tokens used: {e(data.get("tokens_used"))} of {e(data.get("max_total_tokens"))} (+{e(data.get("tokens_this_call"))})</div>'
    if kind == "flag":
        return f'<div class="event warn">{stamp} <b>Flag:</b> {e(data.get("reason"))} {scalar({k: v for k, v in data.items() if k != "reason"})}</div>'
    if kind == "submission_accepted":
        return f'<div class="event good">{stamp} <b>Submission accepted</b> (attempt {e(data.get("attempt"))})</div>'
    if kind == "agent_end":
        return f'<div class="event">{stamp} <b>Agent ended:</b> {e(data.get("status"))} after {e(data.get("turns"))} turns</div>'
    if kind == "workspace_archive":
        contents = data.get("contents", {})
        return (f'<div class="event">{stamp} <b>Workspace archived</b> (not extracted): {e(data.get("bytes"))} bytes, '
                f'{e(contents.get("members"))} members, {e(contents.get("flagged_count"))} flagged · '
                f'{link(folder / "outputs" / (record.get("agent") or "") / str(data.get("file", "")), out_dir, str(data.get("file", "")))}</div>')
    if kind in ("agent_start", "container_stop"):
        return f'<div class="event muted">{stamp} {e(kind)} {scalar(data)}</div>'
    if kind in ("api_error", "api_outage_start", "api_outage_end", "run_error", "workspace_archive_failed", "workspace_archive_incomplete", "unreadable_line"):
        return f'<div class="event bad">{stamp} <b>{e(kind)}</b>{pre(json.dumps(data, indent=1))}</div>'
    # Anything else is shown raw, so nothing in the log is ever left off the page.
    return f'<div class="event">{stamp} <b>{e(kind)}</b>{details("record", pre(json.dumps(data, indent=1)))}</div>'


def agent_summary(records: list[dict]) -> dict:
    """Counts for the overview tables."""
    tool_uses = [b for r in records if r.get("kind") == "api_response"
                 for b in r["data"].get("response", {}).get("content", []) if b.get("type") == "tool_use"]
    budgets = [r["data"].get("tokens_used") for r in records if r.get("kind") == "budget"]
    ends = [r["data"] for r in records if r.get("kind") == "agent_end"]
    return {
        "status": ends[-1].get("status") if ends else "(no agent_end)",
        "turns": sum(1 for r in records if r.get("kind") == "api_response"),
        "run_python calls": sum(1 for b in tool_uses if b.get("name") == "run_python"),
        "submission attempts": sum(1 for b in tool_uses if b.get("name") == "submit_result"),
        "rejections": sum(1 for r in records if r.get("kind") == "rejection"),
        "flags": sum(1 for r in records if r.get("kind") == "flag"),
        "tokens used": budgets[-1] if budgets else None,
    }


def final_submission(records: list[dict]) -> dict | None:
    accepted = [r["data"].get("submission") for r in records if r.get("kind") == "submission_accepted"]
    return accepted[-1] if accepted else None


def render_agent(label: str, records: list[dict], folder: Path, out_dir: Path) -> str:
    by_id = tool_records(records)
    shown_tool_ids = {b.get("id") for r in records if r.get("kind") == "api_response"
                      for b in r["data"].get("response", {}).get("content", []) if b.get("type") == "tool_use"}
    parts = [f'<h2 id="agent-{e(label or "main")}">Agent {e(label or (records[0].get("agent") if records else ""))}</h2>',
             kv(list(agent_summary(records).items())),
             '<p class="muted">Log: ' + link(folder / "log.jsonl", out_dir, "log.jsonl") + " · "
             + link(folder / "outputs", out_dir, "outputs/") + "</p>"]
    submission = final_submission(records)
    if submission is not None:
        parts.append(details("Final accepted submission", show_value(submission), open_=True))
    parts.append("<h3>Timeline</h3>")
    for record in records:
        kind = record.get("kind")
        if kind in ("run_start", "run_end"):
            continue  # shown in the run header
        if kind == "api_response":
            parts.append(render_turn(record, by_id, folder, out_dir))
            continue
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        if kind in TOOL_RECORD_KINDS and data.get("tool_use_id") in shown_tool_ids:
            continue  # shown inside its tool panel
        parts.append(render_event(record, folder, out_dir))
    return "".join(parts)


# ----- across agents (independent runs) ---------------------------------------

def duplicates_across_agents(per_agent: dict[str, list[dict]]) -> str:
    """Identical scripts run by different agents, and identical list-valued or
    long-text fields in their final submissions (e.g. the same seeds). Agents
    that converge on the same code or seeds are not independent measurements."""
    by_hash: dict[str, list[str]] = {}
    for agent, records in per_agent.items():
        for r in records:
            if r.get("kind") == "execution":
                for name, digest in r["data"].get("files", {}).items():
                    if name.endswith(".py"):
                        by_hash.setdefault(digest, []).append(f"{agent}:{name}")
    code_rows = [{"sha256 of script": d[:16] + "…", "run by": ", ".join(where)}
                 for d, where in by_hash.items() if len({w.split(":")[0] for w in where}) > 1]

    field_rows = []
    submissions = {a: final_submission(r) for a, r in per_agent.items()}
    submissions = {a: s for a, s in submissions.items() if isinstance(s, dict)}
    fields = {k for s in submissions.values() for k in s}
    for field in sorted(fields):
        groups: dict[str, list[str]] = {}
        for agent, s in submissions.items():
            value = s.get(field)
            if isinstance(value, (list, dict)) or (isinstance(value, str) and len(value) > 40):
                groups.setdefault(json.dumps(value, sort_keys=True), []).append(agent)
        for value, agents in groups.items():
            if len(agents) > 1:
                field_rows.append({"field": field, "identical in": ", ".join(agents), "value": value[:200]})

    parts = ["<h3>Duplicates across agents</h3>"]
    parts.append(table(code_rows) if code_rows else '<p class="muted">No script was run, byte for byte, by more than one agent.</p>')
    parts.append(table(field_rows) if field_rows else '<p class="muted">No list or long-text submission field is identical across agents.</p>')
    return "".join(parts)


def render_coverage(index_line: dict | None) -> str:
    """Did the analysis grade every accepted submission? (index field analysis_coverage)"""
    if not index_line:
        return ""
    coverage = index_line.get("analysis_coverage")
    if coverage is None:
        return '<p class="muted">Grading coverage not recorded (this run predates the coverage check).</p>'
    if coverage.get("complete") is True:
        return (f'<div class="good">Graded {e(coverage.get("graded"))} of {e(coverage.get("accepted"))} '
                f'accepted submissions.</div>')
    if coverage.get("complete") is False:
        return (f'<div class="bad"><b>Grading incomplete:</b> graded {e(coverage.get("graded"))} of '
                f'{e(coverage.get("accepted"))} accepted submissions. Not graded: '
                f'{e(", ".join(coverage.get("ungraded_agents") or []) or "unknown")}.</div>')
    return f'<div class="warn">Grading coverage unknown: {e(coverage.get("note"))}</div>'


# ----- the whole page ---------------------------------------------------------

STYLE = """
:root{--bg:#fdfdfc;--fg:#1d2329;--muted:#6a737d;--line:#d8dde3;--panel:#f3f5f7;
--good:#e7f4ea;--goodfg:#1d5e2e;--bad:#fbeaea;--badfg:#8b2e2e;--warn:#fff6dd;--accent:#1f4e79;--think:#eef3f8}
@media (prefers-color-scheme: dark){:root{--bg:#15191d;--fg:#dde3e8;--muted:#9aa5b1;--line:#2c343c;
--panel:#1d2329;--good:#16301e;--goodfg:#9fd8ae;--bad:#3a1d1d;--badfg:#f0a8a8;--warn:#3a321a;--accent:#8ab8e6;--think:#1c2733}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5em;margin:.2em 0}h2{border-bottom:2px solid var(--line);padding-top:1em}h3{color:var(--accent)}
h4{margin:.2em 0 .4em;font-size:.95em;color:var(--accent)}
pre{background:var(--panel);padding:8px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;font-size:13px;margin:.3em 0}
table{border-collapse:collapse;font-size:13px;margin:.4em 0}th,td{border:1px solid var(--line);padding:3px 7px;text-align:left;vertical-align:top}
table.kv th{background:var(--panel);font-weight:600;white-space:nowrap}.scroll{overflow-x:auto}
.turn{border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin:12px 0}
.tool{border-left:4px solid var(--accent);padding:4px 10px;margin:8px 0;background:var(--panel)}
.toolhead{font-size:.9em}.thinking{background:var(--think);padding:4px 8px;margin:6px 0}
.purpose{font-size:1.02em;margin:4px 0}
.label{display:inline-block;font-size:.75em;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-right:6px}
.good{background:var(--good);color:var(--goodfg);padding:4px 8px;margin:4px 0}
.bad{background:var(--bad);color:var(--badfg);padding:4px 8px;margin:4px 0}
.warn{background:var(--warn);padding:4px 8px}.event{margin:6px 0;font-size:.93em}.muted{color:var(--muted);font-size:.9em}
.banner{padding:10px 14px;border-radius:6px;font-weight:600;margin:10px 0}
.task{border:1px dashed var(--line);padding:4px 8px;margin:8px 0}a{color:var(--accent)}summary{cursor:pointer}
"""


def render_report(run_dir: Path, out_dir: Path, index_path: Path | None = None) -> str:
    run_dir = Path(run_dir)
    index_line = find_index_line(run_dir, index_path or run_dir.parent / "index.jsonl")
    problems = verification_problems(run_dir)

    if problems:
        rows = [{k: p.get(k) for k in ("log", "kind", "line", "detail") if k in p} for p in problems]
        banner = (f'<div class="banner bad">Verification FAILED: {len(problems)} problem(s). '
                  f'The record below may have been altered; treat it with suspicion.</div>' + table(rows))
    else:
        banner = '<div class="banner good">Verified: every hash chain and stored output file checks out (re-checked when this report was made).</div>'

    folders = agent_folders(run_dir)
    per_agent = {label or "main": read_jsonl(folder / "log.jsonl") for label, folder in folders}
    first_log = next(iter(per_agent.values()), [])
    start = next((r["data"] for r in first_log if r.get("kind") == "run_start"), {})
    metadata = start.get("metadata", {}) if isinstance(start, dict) else {}

    parts = [f"<h1>Run {e(run_dir.name)}</h1>", banner]
    if index_line:
        parts.append(kv([(k, index_line.get(k)) for k in (
            "experiment", "design", "config", "commit", "topology", "model", "effort", "n_agents", "status",
            "agent_statuses", "accepted_submissions", "tokens_used", "wall_clock_s", "timestamp", "final_log_hash")]))
    else:
        parts.append('<div class="banner warn">No line for this run in the index.</div>')
    parts.append(details("Run metadata (from the first log record)", show_value(metadata)))
    for name in ("config.yaml", "design.md", "commit.txt"):
        path = run_dir / name
        if path.is_file():
            parts.append(details(f"{e(name)} (exact copy stored with the run)", pre(path.read_text(encoding="utf-8"))))

    if len(folders) > 1 or (run_dir / "batch.jsonl").exists():
        parts.append("<h2>Across agents</h2>")
        parts.append(table([{"agent": a, **agent_summary(r)} for a, r in per_agent.items()]))
        parts.append(duplicates_across_agents(per_agent))
        batch = read_jsonl(run_dir / "batch.jsonl") if (run_dir / "batch.jsonl").exists() else []
        if batch:
            parts.append(details("batch.jsonl records", "".join(render_event(r, run_dir, out_dir) for r in batch)))

    parts.append("<h2>Analysis</h2>")
    parts.append('<p class="muted">From this run\'s line in runs/index.jsonl. Grading is done after the run '
                 'and is not part of the log; no agent ever saw it.</p>')
    if index_line and index_line.get("analysis_error"):
        parts.append(f'<div class="bad">Analysis error: {e(index_line["analysis_error"])}</div>')
    parts.append(render_coverage(index_line))
    parts.append(show_value(index_line.get("analysis")) if index_line and index_line.get("analysis") is not None
                 else '<p class="muted">No analysis recorded.</p>')

    for label, folder in folders:
        parts.append(render_agent(label, per_agent[label or "main"], folder, out_dir))

    title = f"Run report {run_dir.name}"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{e(title)}</title><style>{STYLE}</style></head><body>{''.join(parts)}</body></html>")


def write_report(run_dir: Path, out_dir: Path = REPORTS_DIR, index_path: Path | None = None) -> Path:
    """Render one run's report into out_dir and return its path."""
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir == run_dir.parent or out_dir.is_relative_to(run_dir.parent) or out_dir.is_relative_to(RUNS_DIR):
        raise ValueError(f"refusing to write a report inside the runs folder ({out_dir})")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_dir.name}.html"
    path.write_text(render_report(run_dir, out_dir, index_path), encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.report",
                                     description="Write an HTML report for each run folder given.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run folder(s), e.g. runs/2026-11-14_...")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR, help="output folder (default: reports/)")
    args = parser.parse_args(argv[1:])
    status = 0
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            print(f"not a folder: {run_dir}", file=sys.stderr)
            status = 2
            continue
        print(write_report(run_dir, args.out))
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
