"""Task definitions for each stage of the screen."""
from . import agent, basic, hard

TASKSETS = {"basic": basic.TASKS, "hard": hard.TASKS}
AGENT_TASKS = agent.TASKS

__all__ = ["TASKSETS", "AGENT_TASKS", "basic", "hard", "agent"]
