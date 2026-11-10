""" The Logger, a hash-chained append-only run log.

Hashes are verified by verify.py.

This code is written and edited by Sean directly. Per invariant 7 in CLAUDE.md, only Sean edits this file. Claude Code is welcome to read this file, call it and of course build tests for it.

## The Log Format

Every entry in the log is canonical JSON of the form:

{"agent", "data", "hash", "kind", "prev_hash", "seq", "ts"}

Note that logs will be in JSONL files, and by fields will appear in alphebetical order.

#### Description of the Fields

* seq: entry number. Natural numbers starting at zero. Gaps or disordered logs are a signal something is off.
* ts: UTC timestamp (ISO 8601).
* agent: who the entry is about (an agent id, or "harness"). 
* kind: the entry type, e.g. tool_call. 
* data: a dict of JSON values.
* prev_hash: previous entry's hash; GENESIS (64 zeros) for the first.
* hash: the canonical JSON of the entry, inclusive of "prev_hash" SHA-256 hashed, but (obviously) exclusive of itself. i.e. the hash field. This is the last thing added. Hence it finishes the link in the hash chain. 


## Rules

* Canonical JSON (sorted keys, no spaces, non-ASCII escaped) gives the same bytes on every device, so the chain can be checked anywhere.
* NaN and infinity are refused: they are not valid JSON. A result that can be undefined or divergent must be represented explicitly by the experiment's submission schema (e.g. a flag plus null).
* Each Logger creates a new file and refuses an existing one, so a finished log can't be reopened and extended.
* each line is flushed and fsynced in case of a crash, so entries elsewhere recorded as written will not be lost.
* .close() never raises an OSError. It always returns the hash of the last entry written, and calling it twice is safe.
* Appends are thread-safe (one lock).
* If a write fails, the logger marks itself broken and refuses every downstream append. The file may end in a partial line, or in a complete line the logger never confirms. Appending after either situation would hide the damage. Hence verify.py reports the first as an incomplete last line and the second as a mismatch with the recorded final hash

## Interface

* Logger(path) : opens a new log with a new file defined by path. Will fail if log already exists.
* .append(agent, kind, data) -> str : appends to the log, and returns the entry's hash
* .last_hash -> str : that's the latest hash written to file.
* .close() -> str : closes the log and returns the final hash. 
"""

import hashlib, json, os, threading
from datetime import datetime, timezone

GENESIS = "0" * 64

def canonical(obj) -> bytes:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    return text.encode("utf-8")

def entry_hash(body: dict) -> str:
    return hashlib.sha256(canonical(body)).hexdigest()


class Logger:
    def __init__(self, path):
        self._file = open(path, "x", encoding="utf-8", newline="\n")
        self._prev = GENESIS
        self._seq = 0
        self._lock = threading.Lock()
        self._broken = False

    def append(self, agent: str, kind: str, data: dict) -> str:   
        with self._lock:
            if self._broken:
                raise RuntimeError("logger is broken after a failed write; refusing to append")
            body = {"seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "agent": agent,
                "kind": kind,
                "data": data,
                "prev_hash": self._prev}
            h = entry_hash(body)
            line = canonical({**body, "hash": h}).decode("utf-8")
            
            try:
                self._file.write(line + "\n")
                self._file.flush()
                os.fsync(self._file.fileno())
            except BaseException:
                self._broken = True
                raise
                
            self._seq += 1
            self._prev = h
            return h

    @property
    def last_hash(self) -> str:
        return self._prev

    def close(self) -> str:
        with self._lock:
            if not self._file.closed:
                try:
                    self._file.close()
                except OSError:
                    self._broken = True  # already a damaged record; verify will report it
        return self._prev
