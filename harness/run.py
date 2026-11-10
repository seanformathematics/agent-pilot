"""Run one experiment config end to end, and record it.

Usage:
    python -m harness.run <path/to/config.yaml>

In order:
  1. Validate the config and everything it points to: prompt files, design,
     submission model, sanity checks. A bad config fails here, before any
     run folder exists. The analysis code is NOT imported yet.
     The logger gets a trial write to a scratch file.
  2. Refuse to start unless the git tree is clean (ignoring runs/), so the
     commit identifies the exact code, prompts, design and config.
  3. Create runs/<date>_<experiment>_<topology>_<nnn>/ holding exact copies
     of the config and design, commit.txt, log.jsonl and outputs/.
  4. Log the setup first: config, commit, Docker image ID, Python versions,
     sandbox packages, SDK version, then every system prompt.
  5. Run the topology. Whatever happens (success, error, Ctrl-C) the
     containers are stopped, the log is closed and its final hash written.
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

from harness.config import REPO_ROOT, load_config, load_object, repo_path
from harness.harness import Harness
from harness.sandbox import Sandbox
from harness.topologies import TOPOLOGIES

RUNS_DIR = REPO_ROOT / "runs"


# ----- before the run ------------------------------------------------------

def check_config_files(config) -> None:
    """Fail early if anything the config points to is missing or won't import."""
    for path in [config.design] + [role.system_prompt for role in config.roles.values()]:
        if not repo_path(path).is_file():
            raise FileNotFoundError(f"config refers to a missing file: {path}")
    load_object(config.submission_model)
    if config.sanity_checks:
        load_object(config.sanity_checks)
    if config.topology not in TOPOLOGIES:
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


def check_logger(logger_factory) -> None:
    """Make sure the logger really writes, using a scratch file, before any
    run folder exists. A run folder can never be deleted, so a broken logger
    must be caught first."""
    with tempfile.TemporaryDirectory(prefix="residency-logger-check-") as scratch:
        path = Path(scratch) / "check.jsonl"
        logger = logger_factory(path)
        logger.append("harness", "logger_check", {})
        logger.close()
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError("the logger wrote nothing; is harness/logger.py implemented?")


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


# ----- the whole run -------------------------------------------------------

def run_experiment(
    config_path: Path,
    client,
    logger_factory,
    sandbox_factory=Sandbox,
    runs_dir: Path = RUNS_DIR,
    get_commit=clean_commit,
    commit_result=commit_run,
) -> dict:
    """Run one config end to end.

    Returns the line written to runs/index.jsonl, plus "git_commit" (the hash
    of the commit holding the run) or "git_commit_error".

    ``logger_factory``, ``sandbox_factory``, ``get_commit`` and
    ``commit_result`` are passed in so tests can supply fakes; the command
    line always uses the real ones.
    """
    config_path = Path(config_path).resolve()
    config_label = (str(config_path.relative_to(REPO_ROOT))
                    if config_path.is_relative_to(REPO_ROOT) else str(config_path))
    config = load_config(config_path)
    check_config_files(config)
    check_logger(logger_factory)
    commit = get_commit()

    run_dir = create_run_dir(Path(runs_dir), config.experiment, config.topology)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    shutil.copyfile(config_path, run_dir / "config.yaml")
    shutil.copyfile(repo_path(config.design), run_dir / "design.md")
    (run_dir / "commit.txt").write_text(commit + "\n")
    (run_dir / "outputs").mkdir()

    logger = logger_factory(run_dir / "log.jsonl")
    work_root = Path(tempfile.mkdtemp(prefix="residency-work-"))
    harness = None
    outcomes = []
    status = "completed"
    try:
        harness = Harness(config, run_dir, work_root, logger, sandbox_factory)
        harness.start_run({
            "run_id": run_dir.name,
            "started_at": started_at,
            "commit": commit,
            "config_path": config_label,
            "sandbox": sandbox_factory.describe_image(config.sandbox.image),
            **host_metadata(),
        })
        if (run_dir / "log.jsonl").stat().st_size == 0:
            raise RuntimeError("the logger wrote nothing; is harness/logger.py implemented?")
        outcomes = TOPOLOGIES[config.topology](harness, client)
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
        (run_dir / "final_hash.txt").write_text(f"{final_hash}\n")
        try:
            work_root.rmdir()  # empty unless a workspace had to be kept
        except OSError:
            pass

    submissions = harness.accepted_submissions if harness else []
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
        "status": status,
        "agent_statuses": {o["agent_id"]: o["status"] for o in outcomes},
        "accepted_submissions": len(submissions),
        "tokens_used": tokens_used,
        "wall_clock_s": wall_clock_s,
        "final_log_hash": final_hash,
        "storage_bytes": {
            "log": (run_dir / "log.jsonl").stat().st_size,
            "outputs": directory_bytes(run_dir / "outputs"),
            "workspace_archives": dict(harness.workspace_archive_bytes) if harness else {},
        },
        **run_analysis(config, submissions),
    }
    index_path = Path(runs_dir) / "index.jsonl"
    append_index_line(index_path, record)

    outcome = {**record, "git_commit": None, "git_commit_error": None}
    try:
        outcome["git_commit"] = commit_result(run_dir, index_path, status)
    except (subprocess.CalledProcessError, OSError) as error:
        detail = getattr(error, "stderr", "") or ""
        outcome["git_commit_error"] = f"{type(error).__name__}: {error} {detail}".strip()

    if status == "aborted":
        raise KeyboardInterrupt
    return outcome


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m harness.run <path/to/config.yaml>", file=sys.stderr)
        return 2
    from harness.agent import make_client
    from harness.logger import Logger

    record = run_experiment(Path(argv[1]), make_client(), Logger)
    print(f"run:        {record['run_id']}")
    print(f"status:     {record['status']}  agents: {record['agent_statuses']}")
    print(f"tokens:     {record['tokens_used']}   wall clock: {record['wall_clock_s']} s")
    print(f"final hash: {record['final_log_hash']}")
    print(f"folder:     {RUNS_DIR / record['run_id']}")
    if record["git_commit"]:
        print(f"committed:  {record['git_commit']}  (not pushed)")
    else:
        print(f"NOT COMMITTED: {record['git_commit_error']}")
        print("              commit the run folder and runs/index.jsonl by hand.")
    return 0 if record["status"] == "completed" and record["git_commit"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
