"""Run one experiment config end to end, and record it.

Usage:
    python -m harness.run <path/to/config.yaml>

In order:
  1. Validate the config and everything it points to: prompt files, design,
     submission model, sanity checks. A bad config fails here, before any
     run folder exists. The analysis code is NOT imported yet.
     The logger and verifier get a trial run on a scratch log.
  2. Refuse to start unless the git tree is clean (ignoring runs/), so the
     commit identifies the exact code, prompts, design and config.
  3. Create runs/<date>_<experiment>_<topology>_<nnn>/ holding exact copies
     of the config and design, and commit.txt.
  4. Run it. One harness means one log ("one harness, one log; folders
     nest the way harnesses nest"):
       single       log.jsonl, outputs/ and final_hash.txt in the run folder.
       independent  one subfolder per agent, each laid out like a single
                    run with its own harness, container and log; plus
                    batch.jsonl and final_hash.txt at the top, recording each
                    agent's final log hash so the batch can't be quietly
                    edited.
     Each log starts with the setup: config, commit, Docker image ID,
     Python versions, sandbox packages, SDK version, then every system
     prompt. Whatever happens (success, error, Ctrl-C) the containers are
     stopped, every log is closed and its final hash written.
  5. Verify: verify.py checks each log and its output files; for
     independent, verify_independent also checks that the subfolders match
     what batch.jsonl recorded.
  6. After the log is closed: import the analysis entry point and call it
     with the accepted submissions, then append one line to runs/index.jsonl.
     This happens for failed and aborted runs too.
  7. Commit exactly two paths to git: this run's folder and runs/index.jsonl.
     Nothing else is staged or committed. Nothing is pushed. If the commit
     fails, the record on disk is untouched and the failure is reported.

The analysis result goes only into the index, never into the log: the log
records what happened during the run, and grading can be redone later
without contradicting it.
"""

import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pydantic

from harness.config import REPO_ROOT, load_config, load_object, load_prompt, repo_path
from harness.harness import Harness
from harness.sandbox import Sandbox
from harness.topologies import TOPOLOGIES, single_agent
from harness.verify import verify

MAX_PROBLEMS_IN_INDEX = 10

# `independent` is a batch of whole runs, so run_independent below runs it,
# not a function in harness/topologies.py.
IMPLEMENTED_TOPOLOGIES = set(TOPOLOGIES) | {"independent"}

# What sits at the top of an independent run folder, besides one subfolder
# per agent. Anything else there is reported by verify_independent.
INDEPENDENT_TOP_LEVEL = {"config.yaml", "design.md", "commit.txt", "batch.jsonl", "final_hash.txt"}

RUNS_DIR = REPO_ROOT / "runs"


# ----- before the run ------------------------------------------------------

def check_config_files(config) -> None:
    """Fail early if anything the config points to is missing or won't import."""
    for path in [config.design] + [role.system_prompt for role in config.roles.values()]:
        if not repo_path(path).is_file():
            raise FileNotFoundError(f"config refers to a missing file: {path}")
    for role in config.roles.values():
        load_prompt(role.system_prompt, config)  # raises on a bad placeholder
    load_object(config.submission_model)
    if config.sanity_checks:
        load_object(config.sanity_checks)
    if config.topology not in IMPLEMENTED_TOPOLOGIES:
        raise NotImplementedError(f"topology {config.topology!r} is not implemented yet")


def clean_commit(repo: Path = REPO_ROOT) -> str:
    """Return the HEAD commit, or refuse if anything outside runs/ is uncommitted.

    runs/ is excluded because finished runs land there between commits;
    nothing in it affects what code, prompts or configs a run used.
    """
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)runs"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    if status.strip():
        raise RuntimeError("git working tree has uncommitted changes; commit them first:\n" + status)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


def check_record_keepers(logger_factory, verifier) -> None:
    """Trial-run the logger and verifier on a scratch log before any run
    folder exists. A run folder can never be deleted, so a broken logger or
    verifier must be caught first. A one-entry log must verify cleanly."""
    with tempfile.TemporaryDirectory(prefix="residency-record-check-") as scratch:
        path = Path(scratch) / "check.jsonl"
        logger = logger_factory(path)
        logger.append("harness", "logger_check", {})
        final_hash = logger.close()
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError("the logger wrote nothing; is harness/logger.py implemented?")
        result = run_verifier(verifier, path, final_hash)
        if not result["ok"]:
            raise RuntimeError(f"the verifier rejected a clean one-entry trial log: {result['problems']}")


def run_verifier(verifier, log_path: Path, final_hash: str) -> dict:
    """Call verify.py on one log and summarise its answer for the index."""
    return summarise_problems(call_verifier(verifier, log_path, final_hash))


def call_verifier(verifier, log_path: Path, final_hash: str | None) -> list[dict]:
    """Call verify.py and return its full list of problems (empty = clean).

    verify(log_path, expected_final_hash) must return (ok, problems), where
    problems is a list of dicts with at least "kind", "line" and "detail".
    A crash or a malformed answer is itself reported as a problem, and so is
    ok=False with no problems listed.
    """
    try:
        answer = verifier(log_path, final_hash)
        if not (isinstance(answer, tuple) and len(answer) == 2
                and isinstance(answer[0], bool) and isinstance(answer[1], list)):
            raise TypeError(f"verify must return (ok, problems); got {answer!r}")
        ok, problems = answer
        if not ok and not problems:
            problems = [{"kind": "verifier_error", "line": None, "detail": "verify said not ok but listed no problems"}]
    except Exception as error:
        problems = [{"kind": "verifier_error", "line": None, "detail": f"{type(error).__name__}: {error}"}]
    return problems


def summarise_problems(problems: list[dict]) -> dict:
    """The index's verify field: ok, how many problems, the first few."""
    return {
        "ok": not problems,
        "problem_count": len(problems),
        "problems": problems[:MAX_PROBLEMS_IN_INDEX],
    }


def create_run_dir(runs_dir: Path, experiment: str, topology: str) -> Path:
    """Make runs/<date>_<experiment>_<topology>_<nnn>/ with the next free number."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = f"{date}_{experiment}_{topology}_"
    runs_dir.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[len(prefix):]) for p in runs_dir.iterdir()
             if p.name.startswith(prefix) and p.name[len(prefix):].isdigit()]
    number = max(taken, default=0) + 1
    while True:
        run_dir = runs_dir / f"{prefix}{number:03d}"
        try:
            run_dir.mkdir()  # fails if it exists, so two runs never share a folder
            return run_dir
        except FileExistsError:
            number += 1


def host_metadata() -> dict:
    return {
        "host_python": sys.version,
        "host_platform": platform.platform(),
        "anthropic_sdk": anthropic.__version__,
        "pydantic": pydantic.__version__,
    }


# ----- after the run -------------------------------------------------------

def commit_run(run_dir: Path, index_path: Path, status: str, repo: Path = REPO_ROOT) -> str:
    """Commit this run's folder and the index, and nothing else. Returns the commit hash.

    `git commit -- <paths>` commits only those paths, even if other changes
    happen to be staged. Your git identity and hooks apply as usual.
    """
    paths = [str(run_dir), str(index_path)]
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", f"run: {run_dir.name} ({status})", "--", *paths],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def run_analysis(config, submissions: list[dict]) -> dict:
    """Import and call the analysis entry point. Errors are recorded, not raised."""
    try:
        analyze = load_object(config.analysis)
        result = analyze(submissions)
        if not isinstance(result, dict) or "estimate" not in result or "passed" not in result:
            raise ValueError(f"analysis must return a dict with 'estimate' and 'passed', got {result!r}")
        return {"analysis": result, "analysis_error": None}
    except Exception as error:
        return {"analysis": None, "analysis_error": f"{type(error).__name__}: {error}"}


def directory_bytes(path: Path) -> int:
    """Total size of the regular files under ``path``, never following links."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            if file_path.is_file() and not file_path.is_symlink():
                total += file_path.stat().st_size
    return total


def append_index_line(index_path: Path, record: dict) -> None:
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----- one harness, one log ------------------------------------------------

def record_one_log(config, folder: Path, client, logger_factory, sandbox_factory,
                   metadata: dict, play) -> dict:
    """Run one harness in ``folder``, recording everything in one hash-chained log.

    ``folder`` gets log.jsonl, outputs/ and final_hash.txt. ``play(harness,
    client)`` runs the agents and returns their outcomes. Whatever happens
    (success, error, Ctrl-C) the containers are stopped, run_end is logged,
    the log is closed and its final hash written. Ctrl-C is not re-raised
    here: the status says "aborted" and the caller decides.

    A single run is one of these in the run folder; an independent run is
    one per agent, each in its own subfolder.
    """
    start = time.monotonic()
    (folder / "outputs").mkdir()
    logger = logger_factory(folder / "log.jsonl")
    harness = None
    outcomes = []
    status = "completed"
    try:
        harness = Harness(config, folder, logger, sandbox_factory)
        harness.start_run(metadata)
        if (folder / "log.jsonl").stat().st_size == 0:
            raise RuntimeError("the logger wrote nothing; is harness/logger.py implemented?")
        outcomes = play(harness, client)
    except KeyboardInterrupt:
        status = "aborted"
    except Exception as error:
        status = "error"
        try:
            logger.append("harness", "run_error", {
                "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        except Exception:
            pass  # the logger itself may be what failed
    finally:
        wall_clock_s = round(time.monotonic() - start, 3)
        tokens_used = harness.tokens_used if harness else 0
        try:
            logger.append("harness", "run_end", {
                "status": status, "agents": outcomes, "tokens_used": tokens_used, "wall_clock_s": wall_clock_s})
        except Exception:
            pass
        final_hash = logger.close()
        (folder / "final_hash.txt").write_text(f"{final_hash}\n")
    return {
        "status": status,
        "outcomes": outcomes,
        "tokens_used": tokens_used,
        "wall_clock_s": wall_clock_s,
        "final_hash": final_hash,
        "submissions": harness.accepted_submissions if harness else [],
        "workspace_archive_bytes": dict(harness.workspace_archive_bytes) if harness else {},
    }


# ----- the two shapes of run folder ----------------------------------------

def run_single_log(config, run_dir: Path, client, logger_factory, sandbox_factory,
                   metadata: dict, verifier) -> dict:
    """single (and later hierarchical): one harness, so one log in the run folder."""
    result = record_one_log(config, run_dir, client, logger_factory, sandbox_factory,
                            metadata, TOPOLOGIES[config.topology])
    return {
        "status": result["status"],
        "outcomes": result["outcomes"],
        "tokens_used": result["tokens_used"],
        "final_hash": result["final_hash"],
        "submissions": result["submissions"],
        "verify": run_verifier(verifier, run_dir / "log.jsonl", result["final_hash"]),
        "storage_bytes": {
            "log": (run_dir / "log.jsonl").stat().st_size,
            "outputs": directory_bytes(run_dir / "outputs"),
            "workspace_archives": result["workspace_archive_bytes"],
        },
        "extra": {},
    }


def run_independent(config, run_dir: Path, client, logger_factory, sandbox_factory,
                    metadata: dict, verifier) -> dict:
    """independent: one fresh harness per agent, one after another.

    Each agent gets its own subfolder, run_dir/<agent id>/, laid out exactly
    like a single run (log.jsonl, outputs/, final_hash.txt), with its own
    harness, container, token budget and hash chain. Nothing is shared
    between them.

    A small top-level log, batch.jsonl, ties the batch together: the config
    and setup, then each agent's status and final log hash as it ends. Its
    own final hash goes in run_dir/final_hash.txt. So deleting, swapping or
    rewriting a whole agent's subfolder is caught, even though that
    subfolder's own log would still verify on its own.

    If one agent's run fails with an error, the next agent still runs: they
    are independent. Ctrl-C stops the batch; agents not yet started are
    logged as not started.
    """
    batch = logger_factory(run_dir / "batch.jsonl")
    status = "completed"
    agent_runs = {}
    try:
        batch.append("harness", "batch_start", {
            "config": config.model_dump(),
            "metadata": metadata,
            "agents": [agent.id for agent in config.agents],
        })
        if (run_dir / "batch.jsonl").stat().st_size == 0:
            raise RuntimeError("the logger wrote nothing; is harness/logger.py implemented?")
        for agent in config.agents:
            if status == "aborted":
                batch.append("harness", "agent_not_started", {"agent_id": agent.id})
                continue
            folder = run_dir / agent.id
            folder.mkdir()
            play = functools.partial(single_agent, agent_id=agent.id)
            result = record_one_log(config, folder, client, logger_factory, sandbox_factory,
                                    {**metadata, "independent_agent": agent.id}, play)
            agent_runs[agent.id] = result
            batch.append("harness", "agent_run_end", {
                "agent_id": agent.id,
                "status": result["status"],
                "final_log_hash": result["final_hash"],
                "tokens_used": result["tokens_used"],
                "wall_clock_s": result["wall_clock_s"],
            })
            if result["status"] == "aborted":
                status = "aborted"
            elif result["status"] == "error" and status == "completed":
                status = "error"
    except KeyboardInterrupt:
        status = "aborted"
    except Exception as error:
        status = "error"
        try:
            batch.append("harness", "batch_error", {
                "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        try:
            batch.append("harness", "batch_end", {
                "status": status,
                "agent_run_statuses": {a: r["status"] for a, r in agent_runs.items()},
                "tokens_used": sum(r["tokens_used"] for r in agent_runs.values()),
            })
        except Exception:
            pass
        final_hash = batch.close()
        (run_dir / "final_hash.txt").write_text(f"{final_hash}\n")

    logs = [run_dir / "batch.jsonl"] + [run_dir / a / "log.jsonl" for a in agent_runs]
    return {
        "status": status,
        "outcomes": [o for r in agent_runs.values() for o in r["outcomes"]],
        "tokens_used": sum(r["tokens_used"] for r in agent_runs.values()),
        "final_hash": final_hash,
        "submissions": [s for r in agent_runs.values() for s in r["submissions"]],
        "verify": summarise_problems(verify_independent(run_dir, final_hash, verifier)),
        "storage_bytes": {
            "log": sum(p.stat().st_size for p in logs if p.is_file()),
            "outputs": sum(directory_bytes(run_dir / a / "outputs") for a in agent_runs),
            "workspace_archives": {a: b for r in agent_runs.values()
                                   for a, b in r["workspace_archive_bytes"].items()},
        },
        "extra": {
            "agent_runs": {
                a: {"status": r["status"], "tokens_used": r["tokens_used"],
                    "wall_clock_s": r["wall_clock_s"], "final_log_hash": r["final_hash"]}
                for a, r in agent_runs.items()
            },
        },
    }


def verify_independent(run_dir: Path, expected_final_hash: str | None, verifier=verify) -> list[dict]:
    """Check a whole independent run folder; return every problem found.

    Uses only Sean's verify() for the hash chains, then checks that the two
    levels agree:
      1. batch.jsonl verifies against expected_final_hash (or, if None, the
         top-level final_hash.txt).
      2. Each agent that batch.jsonl says ended: its subfolder's log
         verifies against the final hash the BATCH LOG recorded, and its
         final_hash.txt says the same.
      3. The run folder holds nothing else: no unrecorded subfolders or files.
    Each problem gets a "log" key naming the log it belongs to. Can be run
    on a finished run folder at any time; never raises.
    """
    run_dir = Path(run_dir)
    batch_log = run_dir / "batch.jsonl"
    try:
        if expected_final_hash is None:
            expected_final_hash = (run_dir / "final_hash.txt").read_text().strip()
    except OSError as error:
        return [{"kind": "file_missing", "line": None, "detail": f"final_hash.txt: {error}",
                 "log": "batch.jsonl"}]
    problems = [{**p, "log": "batch.jsonl"} for p in call_verifier(verifier, batch_log, expected_final_hash)]

    ended = {}  # agent id -> the final log hash recorded in the batch log
    try:
        for number, line in enumerate(batch_log.read_text(encoding="utf-8").splitlines(), start=1):
            entry = json.loads(line)
            if entry.get("kind") == "agent_run_end":
                ended[entry["data"]["agent_id"]] = (entry["data"]["final_log_hash"], number)
    except Exception as error:
        problems.append({"kind": "unreadable_line", "line": None, "log": "batch.jsonl",
                         "detail": f"could not read agent results from batch.jsonl: {type(error).__name__}: {error}"})

    for agent_id, (logged_hash, number) in ended.items():
        label = f"{agent_id}/log.jsonl"
        log_path = run_dir / agent_id / "log.jsonl"
        if not log_path.is_file():
            problems.append({"kind": "file_missing", "line": number, "log": "batch.jsonl",
                             "detail": f"batch.jsonl records agent {agent_id!r}, but {label} is missing"})
            continue
        problems += [{**p, "log": label} for p in call_verifier(verifier, log_path, logged_hash)]
        try:
            written = (run_dir / agent_id / "final_hash.txt").read_text().strip()
        except OSError:
            written = None
        if written != logged_hash:
            problems.append({"kind": "final_hash_mismatch", "line": number, "log": label,
                             "detail": f"{agent_id}/final_hash.txt doesn't match the hash batch.jsonl recorded"})

    if run_dir.is_dir():
        for path in sorted(run_dir.iterdir()):
            expected_dir = path.name in ended and path.is_dir() and not path.is_symlink()
            if path.name not in INDEPENDENT_TOP_LEVEL and not expected_dir:
                problems.append({"kind": "file_extra", "line": None, "log": "batch.jsonl", "file": path.name,
                                 "detail": "in the run folder, but not recorded in batch.jsonl"})
    return problems


# ----- the whole run -------------------------------------------------------

def run_experiment(
    config_path: Path,
    client,
    logger_factory,
    sandbox_factory=Sandbox,
    runs_dir: Path = RUNS_DIR,
    get_commit=clean_commit,
    commit_result=commit_run,
    verifier=verify,
) -> dict:
    """Run one config end to end.

    Returns the line written to runs/index.jsonl, plus "git_commit" (the hash
    of the commit holding the run) or "git_commit_error".

    ``logger_factory``, ``sandbox_factory``, ``get_commit``,
    ``commit_result`` and ``verifier`` are passed in so tests can supply
    fakes; the command line always uses the real ones.
    """
    config_path = Path(config_path).resolve()
    config_label = (str(config_path.relative_to(REPO_ROOT))
                    if config_path.is_relative_to(REPO_ROOT) else str(config_path))
    config = load_config(config_path)
    check_config_files(config)
    check_record_keepers(logger_factory, verifier)
    commit = get_commit()

    run_dir = create_run_dir(Path(runs_dir), config.experiment, config.topology)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    shutil.copyfile(config_path, run_dir / "config.yaml")
    shutil.copyfile(repo_path(config.design), run_dir / "design.md")
    (run_dir / "commit.txt").write_text(commit + "\n")

    metadata = {
        "run_id": run_dir.name,
        "started_at": started_at,
        "commit": commit,
        "config_path": config_label,
        "sandbox": sandbox_factory.describe_image(config.sandbox.image),
        **host_metadata(),
    }
    shape = run_independent if config.topology == "independent" else run_single_log
    result = shape(config, run_dir, client, logger_factory, sandbox_factory, metadata, verifier)
    wall_clock_s = round(time.monotonic() - start, 3)

    record = {
        "run_id": run_dir.name,
        "timestamp": started_at,
        "experiment": config.experiment,
        "design": config.design,
        "config": config_label,
        "commit": commit,
        "topology": config.topology,
        "model": config.model,
        "effort": config.effort,
        "n_agents": len(config.agents),
        "status": result["status"],
        "agent_statuses": {o["agent_id"]: o["status"] for o in result["outcomes"]},
        "accepted_submissions": len(result["submissions"]),
        "tokens_used": result["tokens_used"],
        "wall_clock_s": wall_clock_s,
        "final_log_hash": result["final_hash"],
        "verify": result["verify"],
        "storage_bytes": result["storage_bytes"],
        **result["extra"],
        **run_analysis(config, result["submissions"]),
    }
    index_path = Path(runs_dir) / "index.jsonl"
    append_index_line(index_path, record)

    outcome = {**record, "git_commit": None, "git_commit_error": None}
    try:
        outcome["git_commit"] = commit_result(run_dir, index_path, result["status"])
    except (subprocess.CalledProcessError, OSError) as error:
        detail = getattr(error, "stderr", "") or ""
        outcome["git_commit_error"] = f"{type(error).__name__}: {error} {detail}".strip()

    if result["status"] == "aborted":
        raise KeyboardInterrupt
    return outcome


def check_api_access(client, model: str, environ=os.environ) -> str | None:
    """Before a real run: is there a key, does it work, and does the model exist?

    Returns None if all is well, else a message. Without this, a missing or
    wrong key would only show up on the agent's first API call, after the
    run folder exists, leaving a permanent failed run in runs/. Fetching a
    model's details is free and spends no tokens. Nothing here is logged: no
    run exists yet. The key itself is never printed.
    """
    if not environ.get("ANTHROPIC_API_KEY", "").strip():
        return ("ANTHROPIC_API_KEY is not set in this shell. "
                "Set it in the terminal (never in a file) and try again.")
    try:
        client.models.retrieve(model)
    except anthropic.AuthenticationError:
        return "the API key was rejected (401). Check the key in the Anthropic Console."
    except anthropic.PermissionDeniedError:
        return f"this key may not use model {model!r} (403)."
    except anthropic.NotFoundError:
        return f"model {model!r} was not found (404). Check the model name in the config."
    except anthropic.APIError as error:
        return f"could not reach the API: {type(error).__name__}: {error}"
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m harness.run <path/to/config.yaml>", file=sys.stderr)
        return 2
    from harness.agent import make_client
    from harness.logger import Logger

    client = make_client()
    problem = check_api_access(client, load_config(Path(argv[1])).model)
    if problem:
        print(f"not started (no run folder created): {problem}", file=sys.stderr)
        return 2
    record = run_experiment(Path(argv[1]), client, Logger)
    print(f"run:        {record['run_id']}")
    print(f"status:     {record['status']}  agents: {record['agent_statuses']}")
    print(f"tokens:     {record['tokens_used']}   wall clock: {record['wall_clock_s']} s")
    print(f"final hash: {record['final_log_hash']}")
    print(f"folder:     {RUNS_DIR / record['run_id']}")
    verification = record["verify"]
    if verification["ok"]:
        print("verify:     ok")
    else:
        print(f"VERIFY FAILED: {verification['problem_count']} problem(s); first: {verification['problems'][:3]}")
    if record["git_commit"]:
        print(f"committed:  {record['git_commit']}  (not pushed)")
    else:
        print(f"NOT COMMITTED: {record['git_commit_error']}")
        print("              commit the run folder and runs/index.jsonl by hand.")
    return 0 if record["status"] == "completed" and verification["ok"] and record["git_commit"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
