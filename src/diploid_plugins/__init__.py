"""Built-in state plugins for diploid-agent."""

from diploid_plugins.auto_continue import AutoContinuePlugin
from diploid_plugins.body import BodyPlugin
from diploid_plugins.continuity import ContinuityPlugin
from diploid_plugins.curriculum import CurriculumPlugin
from diploid_plugins.identity import IdentityPlugin
from diploid_plugins.persistent_memory import PersistentMemoryPlugin
from diploid_plugins.planner import PlannerPlugin
from diploid_plugins.self_management import SelfManagementPlugin
from diploid_plugins.self_state import SelfStatePlugin
from diploid_plugins.working_memory import WorkingMemoryPlugin

__all__ = [
    "AutoContinuePlugin",
    "BodyPlugin",
    "ContinuityPlugin",
    "CurriculumPlugin",
    "IdentityPlugin",
    "PersistentMemoryPlugin",
    "PlannerPlugin",
    "SelfManagementPlugin",
    "SelfStatePlugin",
    "WorkingMemoryPlugin",
]
