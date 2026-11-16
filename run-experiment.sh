#!/bin/sh
# Start a harness run: venv interpreter, Mac kept awake for the whole run.

set -e
cd "$(dirname "$0")"
docker info >/dev/null 2>&1 || { echo "Docker is not running"; exit 1; }

caffeinate -i -s .venv/bin/python -m harness.run "$@"
status=$?
osascript -e "display notification \"exit $status\" with title \"Harness run finished\""
exit $status
