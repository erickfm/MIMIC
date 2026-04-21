"""Task/Verifier registry. Task modules register themselves via decorator.

Usage:
    @register_task(MyTask(), MyVerifier())

Then:
    task = get_task("my_task_id")
    verifier = get_verifier("my_task_id")
    all_ids = list_tasks()
    h = registry_hash()   # stamp into artifacts for compat checking
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from rlvr.tasks.base import Task, Verifier

_TASKS: Dict[str, Task] = {}
_VERIFIERS: Dict[str, Verifier] = {}


def register_task(task: Task, verifier: Verifier) -> Tuple[Task, Verifier]:
    """Register one (Task, Verifier) pair. The task's `id` and the
    verifier's `task_id` must match."""
    tid = task.id
    if tid != verifier.task_id:
        raise ValueError(
            f"task.id ({tid!r}) != verifier.task_id ({verifier.task_id!r})"
        )
    if tid in _TASKS:
        raise ValueError(f"task {tid!r} already registered")
    _TASKS[tid] = task
    _VERIFIERS[tid] = verifier
    return task, verifier


def get_task(task_id: str) -> Task:
    return _TASKS[task_id]


def get_verifier(task_id: str) -> Verifier:
    return _VERIFIERS[task_id]


def list_tasks() -> List[str]:
    """All registered task IDs in sorted order (deterministic for hashing)."""
    return sorted(_TASKS.keys())


def registry_hash() -> str:
    """SHA-256 of sorted (id, description) pairs — stamped into
    persisted artifacts for version compatibility."""
    h = hashlib.sha256()
    for tid in list_tasks():
        t = _TASKS[tid]
        h.update(tid.encode())
        h.update(b"\x00")
        h.update(t.description.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _reset_for_testing() -> None:
    """Clear the registry. Only for tests that need to start fresh."""
    _TASKS.clear()
    _VERIFIERS.clear()
