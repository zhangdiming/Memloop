from __future__ import annotations

from collections import Counter

from .store import MemoryStore
from .trace import Tracker


def writer_origin(tr: Tracker, nid: str) -> str:
    st = tr.steps[tr.nodes[nid].step]
    trig = tr.nodes[st.trigger]
    return next(iter(trig.label.origins))


def mem_ancestors(tr: Tracker, nid: str) -> set[str]:
    return {a for a in tr.ancestors([nid]) if tr.nodes[a].kind == "mem"}


def origin_scores(store: MemoryStore, flagged) -> dict[str, dict]:
    tr = store.tracker
    flagged = set(flagged)
    live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
    hit, tot = Counter(), Counter()
    for nid in live:
        writers = {writer_origin(tr, a) for a in mem_ancestors(tr, nid)}
        for o in writers:
            tot[o] += 1
            if nid in flagged:
                hit[o] += 1
    return {o: dict(flagged=hit[o], total=tot[o], lift=hit[o] / tot[o]) for o in tot}


def minimal_flagged(store: MemoryStore, flagged) -> set[str]:
    tr, flagged = store.tracker, set(flagged)
    return {f for f in flagged if not (mem_ancestors(tr, f) - {f}) & flagged}


def source_counts(store: MemoryStore, flagged, blamed: set[str] | None = None) -> Counter:
    tr, flagged = store.tracker, set(flagged)
    count = Counter()
    for f in flagged:
        w = writer_origin(tr, f)
        inherited = [a for a in mem_ancestors(tr, f) - {f} if a in flagged
                     and (o := writer_origin(tr, a)) != w and (blamed is None or o in blamed)]
        if not inherited:
            count[w] += 1
    return count


def responsible_origins(store: MemoryStore, flagged, min_sources: int = 2,
                        bounds: bool = False):
    tr, flagged = store.tracker, set(flagged)
    f = lambda blamed: frozenset(o for o, c in source_counts(store, flagged, blamed).items()
                                 if c >= min_sources)
    lo, hi = frozenset(), f(frozenset())
    while True:
        nlo = f(hi)
        nhi = f(nlo)
        if (nlo, nhi) == (lo, hi):
            break
        lo, hi = nlo, nhi
    return (sorted(lo), sorted(hi)) if bounds else sorted(hi)


def seeds_for_origins(store: MemoryStore, origins) -> list[str]:
    tr, origins = store.tracker, set(origins)
    return [st.trigger for st in tr.steps
            if st.trigger is not None and tr.nodes[st.trigger].label.origins & origins]


def seeds_for_writers(store: MemoryStore, flagged) -> list[str]:
    tr = store.tracker
    return seeds_for_origins(store, {writer_origin(tr, f) for f in flagged if f in tr.nodes})


def seeds_for_entries(store: MemoryStore, flagged) -> list[str]:
    return sorted(set(flagged) & set(store.active_ids()))


def respond(store: MemoryStore, flagged, min_sources: int = 2) -> tuple[str, list[str]]:
    origins = responsible_origins(store, flagged, min_sources)
    if origins:
        return "origin", seeds_for_origins(store, origins)
    return "delete", seeds_for_entries(store, flagged)
