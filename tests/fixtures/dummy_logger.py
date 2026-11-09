"""Stand-in for harness/logger.py, used only by tests until the real one exists.

Same public interface as Sean's skeleton: Logger(path), append(agent, kind, data)
returning a hash string, last_hash, and close(). It writes one JSON line per
entry so tests can read the file back, and also keeps entries in memory.

It does NOT hash or chain anything. The "hashes" are labelled placeholders so
nobody mistakes this for the real thing. Chain tests run against the real
harness/logger.py, never against this.
"""

import json
from datetime import datetime, timezone

GENESIS = "0" * 64


class DummyLogger:
    def __init__(self, path):
        self.path = path
        self._file = open(path, "x", encoding="utf-8")  # refuse to overwrite, like the real one
        self._prev = GENESIS
        self._seq = 0
        self.entries: list[dict] = []

    def append(self, agent: str, kind: str, data: dict) -> str:
        body = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "kind": kind,
            "data": data,
            "prev_hash": self._prev,
        }
        fake_hash = f"dummy-{self._seq:06d}"
        entry = {**body, "hash": fake_hash}
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()
        self.entries.append(entry)
        self._seq += 1
        self._prev = fake_hash
        return fake_hash

    @property
    def last_hash(self) -> str:
        return self._prev

    def close(self) -> str:
        self._file.close()
        return self._prev
