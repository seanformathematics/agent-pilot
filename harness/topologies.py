"""Topologies: one plain-Python function per arrangement of agents in a run.

The orchestrator is never an LLM. These functions dispatch tasks, collect
outcomes and stop on programmatic criteria only. They never interpret
physics and never compare against the known answer.

Each function takes a ready Harness (which owns the sandboxes, the log and
the budget) and an API client, and returns one outcome dict per agent.
Containers are always stopped, even if something fails mid-run.

`independent` (n fresh single-agent runs) is a batch of whole runs, so it
lives with run setup in harness/run.py rather than here (milestone M4).
"""

from harness.agent import Agent


def single(harness, client) -> list[dict]:
    """One agent, end to end."""
    agent_id = harness.config.agents[0].id
    harness.start_agent(agent_id)
    try:
        agent = Agent(agent_id, client, harness)
        outcome = agent.run(harness.task_for(agent_id))
    finally:
        harness.stop_agent(agent_id)
    return [outcome]


# Only topologies that are implemented appear here; run setup refuses others
# before creating a run folder.
TOPOLOGIES = {
    "single": single,
}
