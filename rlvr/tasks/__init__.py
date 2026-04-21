"""Task + verifier registry and the built-in task modules.

Importing this package registers every built-in task. Downstream code
should `import rlvr.tasks` before calling `registry.get_task(...)`.
"""
from rlvr.tasks import l_cancel  # noqa: F401  (import-for-side-effects: registers)
from rlvr.tasks import escape_pressured_shield  # noqa: F401
from rlvr.tasks.base import FrameRow, Prompt, Task, Verifier
from rlvr.tasks.registry import (
    get_task,
    get_verifier,
    list_tasks,
    registry_hash,
)

__all__ = [
    "FrameRow",
    "Prompt",
    "Task",
    "Verifier",
    "get_task",
    "get_verifier",
    "list_tasks",
    "registry_hash",
]
