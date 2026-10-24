import random
from collections import Counter

import pytest

from memloop import HashEmbedder, MemoryStore, Trust, audit_containment, contain, recover
from memloop.recovery import verify_certificate

from test_recovery import KEY, random_script, state, step_fn


def evolve(nb_text: str, new_text: str) -> str | None:
    core, adv = (x.strip() for x in nb_text.split("=>", 1))
    word = new_text.split("=>", 1)[0].split()[-1]
    return None if word in core.split() else f"{core} {word} => {adv}"


def episode(store, query, trig, idx, k):
    hits = store.search(query, k, before=idx)
    prods = step_fn(query, [trig], hits)
    new = next(t for kind, t, _ in prods if kind == "mem")
    nbs = store.search(new, 1, before=idx)
    retrieved = [h.id for h in hits] + [n.id for n in nbs]
    for nb in nbs:
        text = evolve(nb.text, new)
        if text is not None:
            prods.append(("mem", text, {"supersedes": nb.id}))
    return prods, retrieved


def serve(s, session, query, trust, k):
    ctx = s.begin_step(session)
    trig = ctx.observe(query, f"user:{session}", trust, trigger=True)
    prods, retrieved = episode(s, query, trig, ctx.step.index, k)
    ctx.record_retrieval(retrieved)
    for kind, text, meta in prods:
        ctx.emit(kind, text, meta=meta)
    return ctx.step


def execute_evo(script, k):
    s = MemoryStore(HashEmbedder())
    s.tracker.ledger._key = KEY
    triggers = []
    for session, query, trust, evil in script:
        triggers.append((serve(s, session, query, trust, k).trigger, evil))
    return s, triggers


def evo_step_fn(store, k):
    def run(query, obs, hits):
        assert hits is None
        prods, retrieved = episode(store, obs[0].text, obs[0], obs[0].step, k)
        return {"products": prods, "retrieved": retrieved}
    return run


def evo_displaced(store, k):
    def check(step, removed, added):
        tr = store.tracker
        query = tr.nodes[step.trigger].text
        ids = [n.id for n in store.search(query, k, before=step.index)]
        new = [tr.nodes[p] for p in step.products
               if tr.nodes[p].kind == "mem" and not tr.nodes[p].meta.get("supersedes")]
        if new:
            ids += [n.id for n in store.search(new[0].text, 1, before=step.index)]
        return ids != step.retrieved
    return check


def clean_state(script, k, late=()):
    clean, _ = execute_evo([x for x in script if not x[3]] + list(late), k)
    return state(clean)


def test_rewrite_is_versioned():
    s = MemoryStore(HashEmbedder())
    serve(s, "u0", "buy coffee cheap", Trust.USER, 1)
    st = serve(s, "u1", "buy coffee fast", Trust.USER, 1)
    tr = s.tracker
    old, new = [p for p in st.products if tr.nodes[p].kind == "mem"][1:], None
    assert len(old) == 1 and tr.nodes[old[0]].meta["supersedes"] == tr.steps[0].products[0]
    first = tr.steps[0].products[0]
    assert first not in s.active_ids() and s.has(first)
    assert [n.id for n in s.live_before(1)] == [first]
    assert first not in [n.id for n in s.live_before(2)]


@pytest.mark.parametrize("prune", [True, False])
def test_recovery_with_evolution_equals_counterfactual(prune):
    rng = random.Random(11)
    causes, checked, rewrites = Counter(), 0, 0
    for _ in range(200):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_evo(script, k)
        rewrites += sum(1 for n in s.tracker.nodes.values() if n.meta.get("supersedes"))
        cert = recover(s, [t for t, e in triggers if e], evo_step_fn(s, k), prune=prune,
                       displaced=evo_displaced(s, k))
        assert state(s) == clean_state(script, k)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        causes.update(cert.stats["steps_replayed"])
        checked += 1
    assert checked > 150 and rewrites > checked
    if prune:
        assert causes["retrieval"] > 0, causes


def test_contain_restores_clean_versions_and_repair_is_exact():
    rng = random.Random(12)
    checked = restored = 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_evo(script, k)
        tr = s.tracker
        seeds = [t for t, e in triggers if e]
        closure = {n for n in tr.descendants(seeds) if tr.nodes[n].kind == "mem"}
        pre = len(tr.ledger)
        c = contain(s, seeds)
        assert not closure & set(s.active_ids())
        assert audit_containment(tr.ledger.records(), pre, seeds, c["run"]) == []
        restored += sum(r["t"] == "restore" for r in tr.ledger.records()[pre:])
        seen = []
        inner = evo_step_fn(s, k)

        def watched(*a):
            seen.append(bool(closure & set(s.active_ids())))
            return inner(*a)
        cert = recover(s, seeds, watched, displaced=evo_displaced(s, k))
        assert not any(seen)
        assert state(s) == clean_state(script, k)
        assert verify_certificate(cert, tr.ledger.records(), key=KEY) == []
        checked += 1
    assert checked > 100 and restored > 0


def test_tasks_arriving_during_repair_with_evolution():
    rng = random.Random(13)
    checked = 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        late = [(f"u{rng.randint(0, 2)}", f"buy {rng.choice(['coffee', 'shoes'])} "
                 f"{rng.choice(['cheap', 'eco'])}", Trust.USER, False) for _ in range(3)]
        s, triggers = execute_evo(script, k)
        seeds = [t for t, e in triggers if e]
        contain(s, seeds)
        pending, inner = list(late), evo_step_fn(s, k)

        def arriving(*a):
            if pending and rng.random() < 0.5:
                session, query, trust, _ = pending.pop(0)
                serve(s, session, query, trust, k)
            return inner(*a)
        cert = recover(s, seeds, arriving, displaced=evo_displaced(s, k))
        for session, query, trust, _ in pending:
            serve(s, session, query, trust, k)
        assert state(s) == clean_state(script, k, late)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        checked += 1
    assert checked > 100


def test_b3_with_evolution_is_measured():
    rng = random.Random(14)
    wrong = checked = 0
    for _ in range(300):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_evo(script, k)
        recover(s, [t for t, e in triggers if e], evo_step_fn(s, k), displaced=lambda *a: False)
        wrong += state(s) != clean_state(script, k)
        checked += 1
    print(f"B3 with evolution diverged on {wrong}/{checked} scripts")


def args_displaced(store, k):
    tr = store.tracker

    def check(step, removed, added):
        if set(removed) & set(step.retrieved):
            return True
        query = tr.nodes[step.trigger].text
        top = {n.id for n in store.search(query, k, before=step.index)}
        new = [tr.nodes[p] for p in step.products
               if tr.nodes[p].kind == "mem" and not tr.nodes[p].meta.get("supersedes")]
        if not new:
            return bool(added)
        top |= {n.id for n in store.search(new[0].text, 1, before=step.index)}
        return any(n.id in top for n in added)
    return check


@pytest.mark.parametrize("width", [2, 4])
def test_parallel_repair_with_evolution_equals_counterfactual(width):
    rng = random.Random(15)
    checked = 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_evo(script, k)
        cert = recover(s, [t for t, e in triggers if e], evo_step_fn(s, k),
                       displaced=args_displaced(s, k), parallel=width)
        assert state(s) == clean_state(script, k)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        checked += 1
    assert checked > 100
