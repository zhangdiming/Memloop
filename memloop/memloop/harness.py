from __future__ import annotations

from .labels import Trust
from .recovery import StepFn
from .store import MemoryStore
from .trace import Step


def run_step(store: MemoryStore, session: str, query: str, origin: str, trust: Trust,
             step_fn: StepFn, k: int = 3) -> Step:
    ctx = store.begin_step(session)
    trigger = ctx.observe(query, origin, trust, trigger=True)
    hits = ctx.retrieve(query, k) if k else []
    for kind, text, meta in step_fn(query, [trigger], hits):
        ctx.emit(kind, text, meta=meta)
    return ctx.step
