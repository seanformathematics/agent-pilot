"""Harness launcher: runs agent code piped in on stdin as a normal Python script.

Baked into the sandbox image, owned by root and read-only to the agent. For
each run_python call the harness runs, inside the container, as the agent:

    python /opt/harness/launcher.py <exec_id>

with the agent's code on stdin. The launcher saves the code to
/tmp/run_python/<exec_id>.py and runs it as __main__, so that:
  * tracebacks and warnings show the source lines, as in normal Python;
  * __file__ is set, and multiprocessing can start worker processes
    (Python 3.14 starts them by re-importing the main script's file).

The launcher's own frames are removed from tracebacks, so an error looks
exactly as it would if the agent had run `python <file>` itself.
The harness keeps its own copy of the code; this file is only for Python.
"""

import os
import runpy
import sys

CODE_DIR = "/tmp/run_python"


def main() -> None:
    exec_id = sys.argv[1]
    source = sys.stdin.read()

    os.makedirs(CODE_DIR, exist_ok=True)
    path = os.path.join(CODE_DIR, exec_id + ".py")
    if os.path.lexists(path):
        os.remove(path)  # never write through anything planted earlier
    with open(path, "x", encoding="utf-8") as f:
        f.write(source)

    sys.argv = [path]
    sys.excepthook = hide_launcher_frames(path)
    runpy.run_path(path, run_name="__main__")


def hide_launcher_frames(path: str):
    """Print uncaught exceptions starting from the agent's own code."""
    def hook(exc_type, exc, tb):
        while tb is not None and tb.tb_frame.f_code.co_filename != path:
            tb = tb.tb_next
        # The default printer reads the traceback stored on the exception,
        # not the argument, so store the trimmed one there.
        sys.__excepthook__(exc_type, exc.with_traceback(tb), tb)
    return hook


if __name__ == "__main__":
    main()
