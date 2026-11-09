# this file still needs work from sean

import hashlib, json, os, threading
from datetime import datetime, timezone

GENESIS = "0" * 64

def canonical(obj) -> bytes:
    """The one and only way an entry body becomes bytes."""
    ...

def entry_hash(body: dict) -> str:
    """SHA-256 hex of canonical(body). body excludes the 'hash' field."""
    ...

class Logger:
    def __init__(self, path):
        # open with "x"; set self._prev = GENESIS, self._seq = 0, self._lock
        ...

    def append(self, agent: str, kind: str, data: dict) -> str:
        # under the lock: build body (seq, ts, agent, kind, data, prev_hash),
        # compute hash, write one line, flush + fsync,
        # advance seq and prev; return the new hash
        ...

    @property
    def last_hash(self) -> str:
        ...

    def close(self) -> str:
        # close the file, return last_hash for the run index
        ...
