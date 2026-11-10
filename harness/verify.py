"""The verifier: checks a run's hash-chained log and its output files.

Hand-written by Sean. Claude Code may read this file and write tests for it,
but must not edit it (CLAUDE.md, invariant 7). It checks what logger.py writes.

    verify(log_path, expected_final_hash=None) -> (ok, problems)

ok is True exactly when problems is empty. problems lists EVERY problem
found, each a dict: {"kind", "line", "detail"}, plus "file" for file
problems. "line" is the 1-based line in log.jsonl, or None. verify never
raises: anything unexpected becomes a reported problem.

Kinds
-----
    edited_entry          a line's contents don't match its stored hash, or otherwise isn't in the logger's canonical form.
    broken_link           prev_hash or seq doesn't follow the previous line
    unreadable_line       not valid JSON, or not a well-formed entry
    incomplete_last_line  the file doesn't end with a newline
    final_hash_mismatch   the last hash isn't the expected final hash
    file_altered          an output file doesn't match its logged hash
    file_missing          a logged output file doesn't exist
    file_extra            an output file that no log entry mentions

Checks are local: each line is compared with its own stored hash and with
the previous line's STORED hash, so one tampered line is reported once,
not cascaded. After an unreadable line, the next line's link is not
checked. Without expected_final_hash, lines cut off the end are invisible.
"""

import hashlib, json
from pathlib import Path

from harness.logger import GENESIS, canonical, entry_hash

FIELDS = {"seq", "ts", "agent", "kind", "data", "prev_hash", "hash"}


def problem(kind, line, detail, file=None) -> dict:
    p = {"kind": kind, "line": line, "detail": detail}
    if file is not None:
        p["file"] = file
    return p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def well_formed(entry) -> bool:
    return (isinstance(entry, dict) and set(entry) == FIELDS
            and type(entry["seq"]) is int
            and isinstance(entry["agent"], str) and isinstance(entry["kind"], str)
            and isinstance(entry["data"], dict)
            and isinstance(entry["prev_hash"], str) and isinstance(entry["hash"], str))


# ----- the chain -------------------------------------------------------------

def check_chain(log_path: Path, problems: list) -> tuple[str | None, list]:
    """Walk the log. Return the last stored hash and the readable entries."""
    pieces = log_path.read_bytes().split(b"\n")
    last_piece = pieces.pop()  # empty when the file ends with a newline
    if last_piece:
        problems.append(problem("incomplete_last_line", len(pieces) + 1,
                                "the last line has no newline: a write was cut off"))

    entries = []          # (line number, entry)
    prev = GENESIS        # previous line's stored hash; None after a bad line
    next_seq = 0          # None after a bad line
    for number, raw in enumerate(pieces, start=1):
        try:
            entry = json.loads(raw)
            if not well_formed(entry):
                raise ValueError("not a well-formed entry")
            body = {k: v for k, v in entry.items() if k != "hash"}
            if entry_hash(body) != entry["hash"]:
                problems.append(problem("edited_entry", number,
                                        "contents don't match the stored hash"))
            elif raw != canonical(entry):
                problems.append(problem("edited_entry", number,
                                        "line isn't in the logger's exact canonical form"))
        except Exception as error:
            problems.append(problem("unreadable_line", number, f"{type(error).__name__}: {error}"))
            prev = next_seq = None
            continue

        if prev is not None:
            if entry["seq"] != next_seq:
                problems.append(problem("broken_link", number,
                                        f"seq is {entry['seq']}, expected {next_seq}"))
            elif entry["prev_hash"] != prev:
                problems.append(problem("broken_link", number,
                                        "prev_hash doesn't match the previous line's hash"))
        prev = entry["hash"]
        next_seq = entry["seq"] + 1
        entries.append((number, entry))

    return prev, entries


# ----- the output files ------------------------------------------------------

def recorded_files(entries: list, problems: list) -> dict:
    """{(agent, file name): (sha256, line)} from execution and workspace_archive entries."""
    files = {}
    for number, entry in entries:
        try:
            agent, data = entry["agent"], entry["data"]
            if entry["kind"] == "execution":
                named = dict(data.get("files", {}))
            elif entry["kind"] == "workspace_archive":
                named = {data["file"]: data["sha256"],
                         data["file"] + ".err": data["errors_file_sha256"]}
            else:
                continue
            for name, digest in named.items():
                for part in (agent, name):
                    if not isinstance(part, str) or "/" in part or "\\" in part or part in ("", ".", ".."):
                        raise ValueError(f"unsafe path part {part!r}")
                if not isinstance(digest, str):
                    raise ValueError(f"hash for {name!r} is not a string")
                files[(agent, name)] = (digest, number)
        except Exception as error:
            problems.append(problem("unreadable_line", number,
                                    f"bad file record: {type(error).__name__}: {error}"))
    return files


def check_files(outputs: Path, files: dict, problems: list) -> None:
    for (agent, name), (digest, number) in files.items():
        rel = f"{agent}/{name}"
        path = outputs / agent / name
        try:
            if path.is_symlink():
                problems.append(problem("file_altered", number, "is a symlink, not the logged file", rel))
            elif not path.is_file():
                problems.append(problem("file_missing", number, "logged but not present", rel))
            elif sha256_file(path) != digest:
                problems.append(problem("file_altered", number, "contents don't match the logged hash", rel))
        except Exception as error:
            problems.append(problem("file_altered", number, f"could not be read: {error}", rel))

    if not outputs.is_dir():
        return
    for path in sorted(outputs.rglob("*")):
        rel = path.relative_to(outputs)
        if path.is_dir() and not path.is_symlink():
            continue
        if len(rel.parts) == 2 and tuple(rel.parts) in files:
            continue  # a logged file; already checked above
        problems.append(problem("file_extra", None, "not recorded in the log", rel.as_posix()))


# ----- entry point -----------------------------------------------------------

def verify(log_path, expected_final_hash=None) -> tuple[bool, list[dict]]:
    problems = []
    try:
        log_path = Path(log_path)
        last, entries = check_chain(log_path, problems)
        # last is None when the final line was unreadable; that is already reported
        if expected_final_hash is not None and last is not None and last != expected_final_hash:
            problems.append(problem("final_hash_mismatch", None,
                                    "the last hash isn't the recorded final hash "
                                    "(lines added or removed at the end)"))
        check_files(log_path.parent / "outputs", recorded_files(entries, problems), problems)
    except Exception as error:
        problems.append(problem("unreadable_line", None, f"verify could not finish: {type(error).__name__}: {error}"))
    return not problems, problems


if __name__ == "__main__":
    import sys
    ok, found = verify(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print("OK" if ok else "FAILED")
    for p in found:
        where = f"line {p['line']}" if p["line"] else (p.get("file") or "log")
        print(f" - {p['kind']} ({where}): {p['detail']}")
    sys.exit(0 if ok else 1)
