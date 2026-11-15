"""Analysis entry point for the toy experiment.

Only the harness calls this, after the run. The true answer lives here and
nowhere an agent can see it.
"""

TRUE_SUM = 1555


def analyze(submissions: list[dict], agents: list[dict]) -> dict:
    """Grade the run.

    ``submissions`` holds only accepted submissions, in the order they were
    accepted. Each item is
        {"agent_id": str, "attempt": int, "submission": {...validated fields...}}
    Rejected attempts are in the log but are never passed here.

    ``agents`` is every agent in the config, each
        {"agent_id": str, "status": str, "accepted": bool}
    including those that never submitted. The toy grades submissions only and
    just reports the completion count; it never changes "passed".

    Must return at least "estimate" and "passed"; extra keys are allowed.
    Should also return "n_graded" and "graded" (the agent ids it graded), so
    the harness can check that every accepted submission was graded.
    """
    completion = {"accepted": sum(a["accepted"] for a in agents), "agents": len(agents)}
    if not submissions:
        return {"estimate": None, "passed": False, "n_graded": 0, "graded": [], "completion": completion}
    totals = [item["submission"]["total"] for item in submissions]
    # One agent (single) or several (independent): pass only if every
    # accepted answer is right. The estimate is the last one accepted.
    return {"estimate": totals[-1], "passed": all(t == TRUE_SUM for t in totals),
            "n_graded": len(totals), "graded": [item["agent_id"] for item in submissions],
            "completion": completion}


# ----- variants used only by the coverage tests ---------------------------

def analyze_without_coverage(submissions: list[dict], agents: list[dict]) -> dict:
    """An analysis that doesn't report what it graded (the old contract)."""
    return {"estimate": None, "passed": bool(submissions)}


def analyze_first_only(submissions: list[dict], agents: list[dict]) -> dict:
    """A faulty analysis that grades only the first submission."""
    first = submissions[:1]
    return {"estimate": None, "passed": bool(first),
            "n_graded": len(first), "graded": [item["agent_id"] for item in first]}
