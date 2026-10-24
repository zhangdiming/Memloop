from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .ledger import (Ledger, canonical, leaf_hash, verify_consistency, verify_head,
                     verify_inclusion)
from .store import MemoryStore, StepContext, index_digest
from .trace import Node, Step

StepFn = Callable[[str, list[Node], list[Node] | None], list[tuple[str, str, dict]] | dict]
Displaced = Callable[[Step, set[str], list[Node]], bool]

_EPS = 1e-6


@dataclass
class Certificate:
    run: str
    mode: str
    seeds: list[str]
    pre: dict
    post: dict
    consistency: list[str]
    resolutions: list[dict]
    replayed: list[dict]
    dropped: list[int]
    compensations: list[str]
    index_digest: str
    stats: dict = field(default_factory=dict)


def recover(store: MemoryStore, seeds, step_fn: StepFn, mode: str = "quarantine",
            prune: bool = True, run: str | None = None,
            displaced: Displaced | None = None, parallel: int = 1) -> Certificate:
    if mode not in ("quarantine", "delete"):
        raise ValueError(mode)
    tr, ledger = store.tracker, store.tracker.ledger
    seeds = sorted(set(seeds))
    for s in seeds:
        if s not in tr.nodes:
            raise KeyError(s)
    run = run or uuid.uuid4().hex[:12]
    pre = ledger.signed_head()
    tr.log({"t": "recovery_begin", "run": run, "mode": mode, "seeds": seeds,
            "pre_size": pre["size"]})

    res_idx: list[int] = []
    resolved: set[str] = set()
    tainted: set[str] = set()
    removed: set[str] = set()
    added: list[Node] = []
    compensations: list[str] = []
    replayed: list[dict] = []
    dropped: list[int] = []
    searches = checks = 0
    t0 = time.monotonic()

    def resolve(nid: str, action: str) -> None:
        node = tr.nodes[nid]
        res_idx.append(tr.log({"t": "resolve", "run": run, "id": nid, "action": action}))
        resolved.add(nid)
        if node.kind == "mem":
            if store.has(nid):
                store.remove(nid)
            removed.add(nid)
            node.status = "quarantined"
            old = node.meta.get("supersedes")
            if old is not None and old not in tainted and store.has(old):
                if store.retired_by(old) == nid:
                    store.restore(old, run)
                added.append(tr.nodes[old])
        elif node.kind == "obs":
            node.status = "tainted"
        else:
            node.status = "superseded"
        if mode == "delete" and node.kind in ("mem", "obs"):
            tr.erase(nid)

    def resolve_act(nid: str, reproduced: bool) -> None:
        node = tr.nodes[nid]
        if reproduced:
            resolve(nid, "reproduced")
        elif node.meta.get("reversible", True):
            resolve(nid, "undo")
        else:
            resolve(nid, "compensate")
            compensations.append(nid)

    def taint(nid: str) -> None:
        if nid in tainted:
            return
        tainted.add(nid)
        kind = tr.nodes[nid].kind
        if kind != "act":
            resolve(nid, {"obs": "taint", "mem": mode, "out": "retract"}[kind])

    for s in seeds:
        taint(s)

    def may_change(step: Step) -> bool:
        if any(h in removed for h in step.retrieved):
            return True
        cands = [n for n in added if n.step < step.index
                 and (step.min_trust is None or n.label.trust >= step.min_trust)]
        if not cands:
            return False
        if len(step.retrieved) < step.k:
            return True
        kth = tr.nodes[step.retrieved[-1]]
        floor = store.score(step.query, kth.id) - _EPS
        return any(store.score(step.query, n.id) >= floor for n in cands)

    first = min(tr.nodes[s].step for s in seeds)
    write_seeds: dict[int, set[str]] = {}
    for s in seeds:
        if tr.nodes[s].kind == "mem":
            write_seeds.setdefault(tr.nodes[s].step, set()).add(s)
    for i, ss in write_seeds.items():
        new = {p for p in tr.steps[i].products
               if tr.nodes[p].kind == "mem" and not tr.nodes[p].meta.get("supersedes")}
        if new - ss:
            raise ValueError(f"step {i} wrote entries besides the seeded ones; which of its "
                             "re-run entries to suppress is ambiguous")
    seed_steps = {tr.nodes[s].step for s in seeds
                  if tr.nodes[s].kind in ("act", "out")} - set(write_seeds)
    steps = list(tr.steps[first:])
    width = max(1, parallel)
    if width > 1 and (displaced is None or any(st.query is not None for st in steps)):
        raise ValueError("parallel replay needs an agent-side `displaced` check on every step")
    conflicts = 0

    def plan(step: Step, force: str | None = None):
        nonlocal searches, checks
        cause = None
        if step.trigger is not None and step.trigger in tainted:
            cause = "trigger"
        elif any(c in tainted for c in step.context) or any(
                q in tainted for p in step.products for q in tr.nodes[p].parents):
            cause = "context"
        new_hits = step.retrieved
        external = step.query is None and displaced is not None
        if cause != "trigger" and step.query is not None and (
                cause == "context" or not prune or may_change(step)):
            searches += 1
            new_hits = [n.id for n in store.search(step.query, step.k, before=step.index,
                                                   min_trust=step.min_trust)]
            if cause is None and new_hits != step.retrieved:
                cause = "retrieval"
        elif cause is None and external and force is None:
            checks += 1
            if not prune or displaced(step, removed, added):
                cause = "retrieval"
        cause = cause or force
        if cause is None:
            return None

        old = [p for p in step.products if tr.nodes[p].status in ("active", "contained")]
        for p in old:
            taint(p)
        old_acts = [p for p in old if tr.nodes[p].kind == "act"]
        if cause == "trigger":
            step.dropped = True
            step.products = []
            tr.log({"t": "drop_step", "run": run, "step": step.index, "old": old})
            dropped.append(step.index)
            for a in old_acts:
                resolve_act(a, reproduced=False)
            return "dropped"

        query = step.query if step.query is not None else tr.nodes[step.trigger].text
        step.observations = [o for o in step.observations if o not in tainted]
        step.products = []
        obs = [tr.nodes[o] for o in step.observations]
        if external:
            step.retrieved = []
            args = (query, obs, None)
        else:
            step.retrieved = new_hits
            args = (query, obs, [tr.nodes[h] for h in new_hits])
        return dict(step=step, cause=cause, old=old, old_acts=old_acts, args=args)

    def commit(job, produced) -> list[Node]:
        step = job["step"]
        if isinstance(produced, dict):
            if "retrieved" in produced:
                step.retrieved = list(produced["retrieved"])
            produced = dict(produced)
        calls = produced.pop("calls", None) if isinstance(produced, dict) else None
        if isinstance(produced, dict):
            produced = produced["products"]
        suppressed = 0
        if step.index in write_seeds:
            kept = [p for p in produced if p[0] != "mem" or p[2].get("supersedes")]
            suppressed, produced = len(produced) - len(kept), kept
        tr.log({"t": "replay", "run": run, "step": step.index, "cause": job["cause"],
                "hits": step.retrieved, "old": job["old"],
                **({"calls": calls} if calls else {}),
                **({"suppressed": suppressed} if suppressed else {})})
        replayed.append({"step": step.index, "cause": job["cause"], "t": time.monotonic() - t0})
        ctx = StepContext(store, step, replay=True)
        new_nodes = [n for n in (ctx.emit(k, t, meta=m) for k, t, m in produced) if n]
        new_mem = [n for n in new_nodes if n.kind == "mem"]
        added.extend(new_mem)
        removed.update(n.meta["supersedes"] for n in new_mem if n.meta.get("supersedes"))
        new_acts = {n.text for n in new_nodes if n.kind == "act"}
        for a in job["old_acts"]:
            resolve_act(a, reproduced=tr.nodes[a].text in new_acts)
        return new_mem

    run_job = lambda job: step_fn(*job["args"])
    pool = ThreadPoolExecutor(width) if width > 1 else None
    try:
        b = 0
        while b < len(steps) or len(tr.steps) > first + len(steps):
            if len(tr.steps) > first + len(steps):
                late = tr.steps[first + len(steps):]
                if width > 1 and any(st.query is not None for st in late):
                    raise ValueError("parallel replay needs an agent-side `displaced` check on every step")
                steps.extend(late)
            batch, b = steps[b:b + width], b + width
            decided = []
            for step in batch:
                if step.dropped:
                    continue
                if step.index in seed_steps:
                    for p in step.products:
                        if any(q in tainted for q in tr.nodes[p].parents):
                            taint(p)
                    continue
                decided.append((step, plan(step)))
            jobs = [j for _, j in decided if isinstance(j, dict)]
            outs = list(pool.map(run_job, jobs)) if pool and len(jobs) > 1 else \
                [run_job(j) for j in jobs]
            result = {id(j): o for j, o in zip(jobs, outs)}
            fresh: list[Node] = []
            fresh_rm: set[str] = set()
            for step, job in decided:
                if job == "dropped":
                    continue
                seen = fresh or fresh_rm
                if job is None:
                    if not (seen and displaced(step, fresh_rm, fresh)):
                        continue
                    conflicts += 1
                    job = plan(step, force="retrieval")
                    produced = run_job(job)
                else:
                    produced = result[id(job)]
                    if isinstance(produced, dict) and "retrieved" in produced:
                        step.retrieved = list(produced["retrieved"])
                    if seen and step.query is None and displaced(step, fresh_rm, fresh):
                        conflicts += 1
                        produced = run_job(job)
                new_mem = commit(job, produced)
                fresh += new_mem
                fresh_rm.update(n.meta["supersedes"] for n in new_mem if n.meta.get("supersedes"))
    finally:
        if pool:
            pool.shutdown()

    for nid in sorted(tainted - resolved):
        resolve_act(nid, reproduced=False)

    digest = store.index_digest()
    tr.log({"t": "recovery_end", "run": run, "index_digest": digest})
    post = ledger.signed_head()

    closure = Counter(tr.nodes[n].kind for n in tainted)
    stats = {
        "closure": dict(closure),
        "sessions_affected": len({tr.steps[tr.nodes[n].step].session for n in tainted}),
        "steps_after_tau": len(tr.steps) - first,
        "steps_replayed": Counter(r["cause"] for r in replayed),
        "steps_dropped": len(dropped),
        "searches": searches,
        "displacement_checks": checks,
        "parallel": width,
        "parallel_conflicts": conflicts,
        "compensations": len(compensations),
    }
    return Certificate(
        run=run, mode=mode, seeds=seeds, pre=pre, post=post,
        consistency=[h.hex() for h in ledger.consistency_proof(pre["size"], post["size"])],
        resolutions=[{"index": i, "record": ledger.record(i),
                      "proof": [h.hex() for h in ledger.inclusion_proof(i, post["size"])]}
                     for i in res_idx],
        replayed=replayed, dropped=dropped, compensations=compensations,
        index_digest=digest, stats=stats)


def contain(store: MemoryStore, seeds, run: str | None = None) -> dict:
    tr, ledger = store.tracker, store.tracker.ledger
    t0 = time.monotonic()
    run = run or uuid.uuid4().hex[:12]
    pre = ledger.signed_head()
    seeds = sorted(set(seeds))
    tr.log({"t": "contain_begin", "run": run, "seeds": seeds, "pre_size": pre["size"]})
    contained = []
    closure = tr.descendants(seeds)
    for nid in sorted(closure, key=lambda n: tr.nodes[n].order):
        node = tr.nodes[nid]
        if node.kind == "mem" and store.has(nid):
            store.remove(nid)
            node.status = "contained"
            tr.log({"t": "contain", "run": run, "id": nid})
            contained.append(nid)
            old = node.meta.get("supersedes")
            if old is not None and old not in closure and store.retired_by(old) == nid:
                store.restore(old, run)
    digest = store.index_digest()
    tr.log({"t": "contain_end", "run": run, "index_digest": digest})
    post = ledger.signed_head()
    return dict(run=run, seeds=seeds, contained=contained, pre=pre, post=post,
                consistency=[h.hex() for h in ledger.consistency_proof(pre["size"], post["size"])],
                index_digest=digest, seconds=time.monotonic() - t0)


def audit_containment(records: list[dict], pre_size: int, seeds, run: str) -> list[str]:
    children: dict[str, list[str]] = {}
    kind: dict[str, str] = {}
    for r in records[:pre_size]:
        if r["t"] == "node":
            kind[r["id"]] = r["kind"]
            for p in r["parents"]:
                children.setdefault(p, []).append(r["id"])
    gone = {r["id"] for r in records[:pre_size] if r["t"] in ("resolve", "contain")}
    done = {r["id"] for r in records[pre_size:] if r.get("run") == run and r["t"] == "contain"}
    closure, stack = set(seeds), list(seeds)
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in closure:
                closure.add(c)
                stack.append(c)
    return [f"derived entry {n} still live after containment" for n in sorted(closure)
            if kind.get(n) == "mem" and n not in gone and n not in done]


def attribute(store: MemoryStore, flagged) -> list[Node]:
    roots = store.tracker.sources(flagged)
    low = min((n.label.trust for n in roots), default=None)
    return [n for n in roots if n.label.trust == low]


def audit(records: list[dict], pre_size: int, seeds, run: str) -> list[str]:
    problems = []
    children: dict[str, list[str]] = {}
    for r in records[:pre_size]:
        if r["t"] == "node":
            for p in r["parents"]:
                children.setdefault(p, []).append(r["id"])
    resolved_before = {r["id"] for r in records[:pre_size] if r["t"] == "resolve"}
    mine = [r for r in records[pre_size:] if r.get("run") == run]
    resolved = {r["id"] for r in mine if r["t"] == "resolve"}

    closure, stack = set(seeds), list(seeds)
    while stack:
        for c in children.get(stack.pop(), ()):
            if c not in closure and c not in resolved_before:
                closure.add(c)
                stack.append(c)
    for nid in sorted(closure - resolved_before - resolved):
        problems.append(f"descendant {nid} of the seeds was never resolved")
    held: dict[int, set[str]] = {}
    gone = set(resolved_before)
    for r in records:
        if r["t"] == "node" and r["kind"] != "obs":
            held.setdefault(r["step"], set()).add(r["id"])
        elif r["t"] == "resolve":
            gone.add(r["id"])
        elif r["t"] in ("replay", "drop_step") and r.get("run") == run:
            due = held.get(r["step"], set()) - gone
            for nid in sorted(due - set(r["old"])):
                problems.append(f"step {r['step']}: product {nid} missing from the replay record")
            for nid in sorted(due | set(r["old"])):
                if nid not in resolved:
                    problems.append(f"step {r['step']} re-ran but its product {nid} was not resolved")
            held[r["step"]] = set()
    return problems


def replay_index_digest(records: list[dict]) -> str:
    live: dict[str, str] = {}
    retired: dict[str, str] = {}
    for r in records:
        if r["t"] == "node" and r["kind"] == "mem":
            live[r["id"]] = r["vec"]
        elif r["t"] == "retire" and r["id"] in live:
            retired[r["id"]] = live.pop(r["id"])
        elif r["t"] == "restore" and r["id"] in retired:
            live[r["id"]] = retired.pop(r["id"])
        elif r["t"] in ("resolve", "contain"):
            live.pop(r["id"], None)
            retired.pop(r["id"], None)
    return index_digest(live.items())


def verify_certificate(cert: Certificate, records: list[dict] | None = None,
                       key: bytes | None = None) -> list[str]:
    problems = []
    pre, post = cert.pre, cert.post
    if key is not None and not (verify_head(pre, key) and verify_head(post, key)):
        problems.append("tree head signature invalid")
    if not verify_consistency(pre["size"], post["size"], bytes.fromhex(pre["root"]),
                              bytes.fromhex(post["root"]),
                              [bytes.fromhex(h) for h in cert.consistency]):
        problems.append("log was not append-only between the two heads")
    for res in cert.resolutions:
        if not pre["size"] <= res["index"] < post["size"]:
            problems.append(f"resolution at {res['index']} is outside the recovery window")
        if not verify_inclusion(leaf_hash(canonical(res["record"])), res["index"], post["size"],
                                [bytes.fromhex(h) for h in res["proof"]],
                                bytes.fromhex(post["root"])):
            problems.append(f"resolution at {res['index']} is not in the log")
    if records is not None:
        log = Ledger()
        for r in records[:post["size"]]:
            log.append(r)
        if len(records) < post["size"] or log.root().hex() != post["root"]:
            problems.append("supplied records do not match the post-recovery head")
            return problems
        problems += audit(records, pre["size"], cert.seeds, cert.run)
        if replay_index_digest(records[:post["size"]]) != cert.index_digest:
            problems.append("index digest does not match the live set implied by the log")
    return problems
