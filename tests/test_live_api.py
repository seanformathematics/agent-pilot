"""M3: one real toy run against the Anthropic API. Costs real money.

Skipped by default (pyproject.toml adds -m 'not api'). Run only with Sean's
go-ahead:

    .venv/bin/python -m pytest -m api tests/test_live_api.py -v

It writes into a temporary folder, not runs/, and commits nothing. The
recorded M3 run itself goes through the command line:

    .venv/bin/python -m harness.run tests/fixtures/toy_sum/config.yaml
"""

import os

import pytest

from harness.agent import make_client
from harness.config import load_config
from harness.logger import Logger
from harness.run import check_api_access, run_experiment
from harness.sandbox import Sandbox
from tests.conftest import TOY_CONFIG

pytestmark = [
    pytest.mark.api,
    pytest.mark.docker,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


def test_live_toy_run(tmp_path):
    client = make_client()
    problem = check_api_access(client, load_config(TOY_CONFIG).model)
    assert problem is None, problem

    record = run_experiment(TOY_CONFIG, client, Logger, sandbox_factory=Sandbox,
                            runs_dir=tmp_path / "runs", get_commit=lambda: "live-test",
                            commit_result=lambda run_dir, index_path, status: "not-committed")

    print(f"\nstatus {record['status']}, agents {record['agent_statuses']}, "
          f"tokens {record['tokens_used']}, {record['wall_clock_s']} s, "
          f"analysis {record['analysis']}")
    assert record["status"] == "completed"
    assert record["verify"]["ok"], record["verify"]
