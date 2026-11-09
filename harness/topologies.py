"""Topologies: one plain-Python function per arrangement of agents.

The orchestrator is never an LLM. These functions dispatch tasks, collect
results and stop on programmatic criteria only. They never interpret physics
and never compare against the known answer.
"""


def single(config: dict, run_dir, client) -> dict:
    """One agent, end to end."""
    raise NotImplementedError


def independent(config: dict, run_dir, client) -> dict:
    """n separate single-agent runs of the same task, each with a fresh harness."""
    raise NotImplementedError


def one_shot(config: dict, run_dir, client) -> dict:
    """Fixed task specs to isolated workers; one input, one output each."""
    raise NotImplementedError


def hierarchical(config: dict, run_dir, client) -> dict:
    """A supervisor directs workers via send_message. No sideways messages."""
    raise NotImplementedError


TOPOLOGIES = {
    "single": single,
    "independent": independent,
    "one_shot": one_shot,
    "hierarchical": hierarchical,
}
