"""Sanity checks for the toy experiment. They never look at the true answer."""


def check_submission(submission, task: str) -> list[str]:
    problems = []
    if not submission.method_notes.strip():
        problems.append("method_notes is empty; describe how the sum was computed.")
    return problems
