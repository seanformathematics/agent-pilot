# This file still needs work from sean.

def verify(path, expected_final_hash=None) -> tuple[bool, str]:
    """Walk the log; check each hash, each prev_hash link and seq order.
    Optionally compare the last hash to the recorded one.
    Return (ok, message naming the first broken line if any)."""
    ...
