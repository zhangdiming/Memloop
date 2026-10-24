import random
from collections import Counter

import pytest

from memloop import (HashEmbedder, MemoryStore, Trust, attribute, audit, audit_containment,
                     contain, recover, run_step)
from memloop.recovery import verify_certificate

KEY = b"auditor-shared-key"
TOPICS = ["toothbrush", "laptop", "coffee", "shoes"]
MODS = ["cheap", "fast", "best", "eco", "small"]


def step_fn(query, obs, hits):
    q = obs[0].text if obs else query
    if "=>" in q:
        core, adv = (x.strip() for x in q.split("=>", 1))
    else:
        core = q.strip()
        adv = hits[0].text.split("=>", 1)[1].strip() if hits else core.split()[-1]
    return [("mem", f"{core} => {adv}", {}),
            ("act", f"buy({adv})", {"reversible": False}),
            ("out", adv, {})]


def random_script(rng):
    script = []
    for i in range(rng.randint(6, 24)):
        topic, mod = rng.choice(TOPICS), rng.choice(MODS)
        if rng.random() < 0.2:
            script.append(("mallory", f"buy {topic} {mod} => EVIL{i}", Trust.UNTRUSTED, True))
        else:
            script.append((f"u{rng.randint(0, 2)}", f"buy {topic} {mod}", Trust.USER, False))
    return script


def execute(script, k):
    s = MemoryStore(HashEmbedder())
    s.tracker.ledger._key = KEY
    triggers = []
    for session, query, trust, evil in script:
        st = run_step(s, session, query, f"user:{session}", trust, step_fn, k=k)
        triggers.append((st.trigger, evil))
    return s, triggers


def state(s):
    tr = s.tracker
    mem = [tr.nodes[i].text for i in sorted(s.active_ids(), key=lambda i: tr.nodes[i].order)]
    steps = [(st.session, [(tr.nodes[p].kind, tr.nodes[p].text) for p in st.products])
             for st in tr.steps if not st.dropped]
    return mem, steps


@pytest.mark.parametrize("prune", [True, False])
def test_recovery_equals_counterfactual(prune):
    rng = random.Random(0)
    causes, checked = Counter(), 0
    for _ in range(300):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute(script, k)
        seeds = [t for t, evil in triggers if evil]
        cert = recover(s, seeds, step_fn, prune=prune)
        clean, _ = execute([x for x in script if not x[3]], k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        assert cert.index_digest == s.index_digest()
        causes.update(cert.stats["steps_replayed"])
        checked += 1
    assert checked > 200
    assert causes["retrieval"] > 0, causes


def test_recovering_twice_equals_recovering_once():
    rng = random.Random(7)
    checked = 0
    for _ in range(200):
        script, k = random_script(rng), rng.randint(1, 3)
        s, triggers = execute(script, k)
        evil = [t for t, e in triggers if e]
        if len(evil) < 2:
            continue
        first, second = evil[::2], evil[1::2]
        recover(s, first, step_fn)
        cert = recover(s, second, step_fn)
        clean, _ = execute([x for x in script if not x[3]], k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        checked += 1
    assert checked > 50


def test_pruning_does_less_work_for_same_result():
    rng = random.Random(1)
    full = pruned = 0
    for _ in range(100):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        a, ta = execute(script, k)
        b, tb = execute(script, k)
        ca = recover(a, [t for t, e in ta if e], step_fn, prune=False)
        cb = recover(b, [t for t, e in tb if e], step_fn, prune=True)
        assert state(a) == state(b)
        full += ca.stats["searches"]
        pruned += cb.stats["searches"]
    assert pruned < full


def scenario():
    script = [
        ("u0", "buy toothbrush cheap", Trust.USER, False),
        ("mallory", "buy buy toothbrush toothbrush cheap cheap => EVILBRAND", Trust.UNTRUSTED, True),
        ("u1", "buy toothbrush cheap", Trust.USER, False),
        ("u2", "order laptop fast", Trust.USER, False),
        ("u1", "buy toothbrush cheap", Trust.USER, False),
    ]
    return script, *execute(script, 1)


def test_writer_seeds_stop_a_relay_of_fresh_accounts():
    from memloop.attribution import responsible_origins, seeds_for_writers
    rng = random.Random(11)
    checked = 0
    for _ in range(200):
        script = [(f"m{i}" if e else sess, q, t, e) for i, (sess, q, t, e) in enumerate(random_script(rng))]
        k = rng.randint(1, 3)
        s, _ = execute(script, k)
        tr = s.tracker
        flagged = [i for i in s.active_ids() if tr.nodes[i].kind == "mem"
                   and tr.steps[tr.nodes[i].step].session.startswith("m")]
        if len(flagged) < 2:
            continue
        clean, _ = execute([x for x in script if not x[3]], k)
        assert len(responsible_origins(s, flagged)) == 0
        cert = recover(s, seeds_for_writers(s, flagged), step_fn)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        checked += 1
    assert checked > 50


def test_respond_falls_back_to_deletion_without_replay():
    from memloop.attribution import respond
    rng = random.Random(5)
    seen = Counter()
    for _ in range(300):
        script = random_script(rng)
        relay = rng.random() < 0.5
        if relay:
            script = [(f"m{i}" if e else sess, q, t, e) for i, (sess, q, t, e) in enumerate(script)]
        k = rng.randint(1, 3)
        s, _ = execute(script, k)
        tr = s.tracker
        evil = {sess for sess, _, _, e in script if e}
        flagged = [i for i in s.active_ids() if tr.nodes[i].kind == "mem"
                   and tr.steps[tr.nodes[i].step].session in evil]
        if len(flagged) < 2:
            continue
        how, ids = respond(s, flagged)
        if relay:
            assert (how, ids) == ("delete", sorted(flagged))
        else:
            assert how == "origin"
            clean, _ = execute([x for x in script if not x[3]], k)
            recover(s, ids, step_fn)
            assert state(s) == state(clean)
        seen[how] += 1
    assert seen["origin"] > 30 and seen["delete"] > 30


def test_attack_spreads_then_attribution_and_recovery():
    script, s, triggers = scenario()
    tr = s.tracker
    victim = tr.steps[2]
    assert any(tr.nodes[p].text == "EVILBRAND" for p in victim.products)

    flagged = [p for p in victim.products if tr.nodes[p].kind == "mem"]
    roots = attribute(s, flagged)
    assert [r.id for r in roots] == [triggers[1][0]]

    cert = recover(s, [r.id for r in roots], step_fn)
    assert cert.dropped == [1]
    assert len(cert.compensations) == 3
    assert cert.stats["sessions_affected"] == 2
    clean, _ = execute([x for x in script if not x[3]], 1)
    assert state(s) == state(clean)


def test_audit_catches_skipped_resolution():
    _, s, triggers = scenario()
    cert = recover(s, [triggers[1][0]], step_fn)
    records = s.tracker.ledger.records()
    target = next(r for r in records[cert.pre["size"]:] if r["t"] == "resolve"
                  and r["action"] == "quarantine")
    tampered = [r for r in records if r is not target]
    assert audit(tampered, cert.pre["size"], cert.seeds, cert.run)
    assert verify_certificate(cert, tampered, key=KEY)


def test_ghost_vector_detected():
    _, s, triggers = scenario()
    cert = recover(s, [triggers[1][0]], step_fn)
    gone = cert.resolutions[1]["record"]["id"]
    s._insert(gone, s.embedder([s.tracker.nodes[gone].text])[0])
    assert s.index_digest() != cert.index_digest


def test_verified_deletion_of_one_entry():
    _, s, triggers = scenario()
    tr = s.tracker
    target = next(p for p in tr.steps[0].products if tr.nodes[p].kind == "mem")
    text = tr.nodes[target].text
    cert = recover(s, [target], step_fn, mode="delete")
    assert verify_certificate(cert, tr.ledger.records(), key=KEY) == []
    assert tr.nodes[target].text is None and tr.nodes[target].salt is None
    assert target not in s.active_ids()
    assert all(tr.nodes[i].step != 0 for i in s.active_ids())
    assert text not in str(tr.ledger.records())
    assert any(r["step"] == 3 and r["cause"] == "context" for r in cert.replayed)


def execute_external(script, k, fn=None):
    fn = fn or step_fn
    s = MemoryStore(HashEmbedder())
    s.tracker.ledger._key = KEY
    triggers = []
    for session, query, trust, evil in script:
        ctx = s.begin_step(session)
        trig = ctx.observe(query, f"user:{session}", trust, trigger=True)
        hits = s.search(query, k, before=ctx.step.index)
        ctx.record_retrieval([h.id for h in hits])
        for kind, text, meta in fn(query, [trig], hits):
            ctx.emit(kind, text, meta=meta)
        triggers.append((ctx.step.trigger, evil))
    return s, triggers


def external_step_fn(store, k, fn=None):
    fn = fn or step_fn

    def run(query, obs, hits):
        assert hits is None
        idx = obs[0].step
        hits = [n for n in store.search(obs[0].text, k, before=idx)]
        return {"products": fn(query, obs, hits), "retrieved": [h.id for h in hits]}
    return run


def external_displaced(store, k):
    def check(step, removed, added):
        query = store.tracker.nodes[step.trigger].text
        return [n.id for n in store.search(query, k, before=step.index)] != step.retrieved
    return check


@pytest.mark.parametrize("prune", [True, False])
def test_external_retrieval_recovery_equals_counterfactual(prune):
    rng = random.Random(2)
    causes, checked = Counter(), 0
    for _ in range(200):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_external(script, k)
        cert = recover(s, [t for t, e in triggers if e], external_step_fn(s, k), prune=prune,
                       displaced=external_displaced(s, k))
        clean, _ = execute_external([x for x in script if not x[3]], k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        causes.update(cert.stats["steps_replayed"])
        checked += 1
    assert checked > 150
    if prune:
        assert causes["retrieval"] > 0, causes


def watched(store, step_fn, closure, log):
    def run(query, obs, hits):
        log.append(any(n in closure for n in store.active_ids()))
        return step_fn(query, obs, hits)
    return run


@pytest.mark.parametrize("external", [False, True])
def test_contain_then_repair_is_safe_throughout_and_exact(external):
    rng = random.Random(3)
    unsafe_plain = checked = 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        ex = execute_external if external else execute
        mk = (lambda s: external_step_fn(s, k)) if external else (lambda s: step_fn)
        disp = (lambda s: external_displaced(s, k)) if external else (lambda s: None)

        s, triggers = ex(script, k)
        seeds = [t for t, e in triggers if e]
        closure = {n for n in s.tracker.descendants(seeds) if s.tracker.nodes[n].kind == "mem"}
        pre = len(s.tracker.ledger)
        c = contain(s, seeds)
        assert not closure & set(s.active_ids())
        assert audit_containment(s.tracker.ledger.records(), pre, seeds, c["run"]) == []
        seen = []
        cert = recover(s, seeds, watched(s, mk(s), closure, seen), displaced=disp(s))
        assert not any(seen), "a contaminated entry was live during the repair"
        clean, _ = ex([x for x in script if not x[3]], k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []

        p, triggers = ex(script, k)
        seen = []
        recover(p, [t for t, e in triggers if e], watched(p, mk(p), closure, seen),
                displaced=disp(p))
        unsafe_plain += any(seen)
        checked += 1
    assert checked > 100
    assert unsafe_plain > 0


@pytest.mark.parametrize("external", [False, True])
def test_tasks_served_during_repair_are_repaired_too(external):
    rng = random.Random(4)
    causes, checked = Counter(), 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        late = [(f"u{rng.randint(0, 2)}", f"buy {rng.choice(TOPICS)} {rng.choice(MODS)}",
                 Trust.USER, False) for _ in range(rng.randint(1, 5))]
        ex = execute_external if external else execute
        s, triggers = ex(script, k)
        seeds = [t for t, e in triggers if e]
        closure = {n for n in s.tracker.descendants(seeds) if s.tracker.nodes[n].kind == "mem"}
        contain(s, seeds)
        for session, query, trust, _ in late:
            if external:
                ctx = s.begin_step(session)
                trig = ctx.observe(query, f"user:{session}", trust, trigger=True)
                hits = s.search(query, k, before=ctx.step.index)
                assert not closure & {h.id for h in hits}
                ctx.record_retrieval([h.id for h in hits])
                for kind, text, meta in step_fn(query, [trig], hits):
                    ctx.emit(kind, text, meta=meta)
            else:
                st = run_step(s, session, query, f"user:{session}", trust, step_fn, k=k)
                assert not closure & set(st.retrieved)
        cert = recover(s, seeds, external_step_fn(s, k) if external else step_fn,
                       displaced=external_displaced(s, k) if external else None)
        clean, _ = ex([x for x in script if not x[3]] + late, k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        causes.update(cert.stats["steps_replayed"])
        checked += 1
    assert checked > 100
    assert causes["retrieval"] > 0, causes


@pytest.mark.parametrize("external", [False, True])
def test_tasks_arriving_while_repair_runs_are_repaired_too(external):
    rng = random.Random(5)
    checked = arrived_total = 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        late = [(f"u{rng.randint(0, 2)}", f"buy {rng.choice(TOPICS)} {rng.choice(MODS)}",
                 Trust.USER, False) for _ in range(rng.randint(1, 5))]
        ex = execute_external if external else execute
        s, triggers = ex(script, k)
        seeds = [t for t, e in triggers if e]
        contain(s, seeds)
        pending = list(late)
        inner = external_step_fn(s, k) if external else step_fn

        def serve(session, query, trust):
            if external:
                ctx = s.begin_step(session)
                trig = ctx.observe(query, f"user:{session}", trust, trigger=True)
                hits = s.search(query, k, before=ctx.step.index)
                ctx.record_retrieval([h.id for h in hits])
                for kind, text, meta in step_fn(query, [trig], hits):
                    ctx.emit(kind, text, meta=meta)
            else:
                run_step(s, session, query, f"user:{session}", trust, step_fn, k=k)

        def arriving(*args):
            if pending and rng.random() < 0.5:
                session, query, trust, _ = pending.pop(0)
                serve(session, query, trust)
            return inner(*args)

        cert = recover(s, seeds, arriving,
                       displaced=external_displaced(s, k) if external else None)
        arrived = len(late) - len(pending)
        if not arrived:
            continue
        for session, query, trust, _ in pending:
            serve(session, query, trust)
        clean, _ = ex([x for x in script if not x[3]] + late, k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        checked += 1
        arrived_total += arrived
    assert checked > 50 and arrived_total > checked


def test_attribution_of_accounts_that_copy_each_other_is_stable():
    from memloop.attribution import responsible_origins
    s = MemoryStore(HashEmbedder())

    def add(origin, text, hits):
        ctx = s.begin_step(origin)
        ctx.observe(text, "user:" + origin, Trust.USER, trigger=True)
        ctx.record_retrieval(hits)
        return ctx.emit("mem", text).id

    a1 = add("A", "a1", [])
    b1 = add("B", "b1", [])
    a2 = add("A", "a2", [b1])
    b2 = add("B", "b2", [a1])
    v = add("V", "v", [a2])
    flagged = [a1, b1, a2, b2, v]
    assert responsible_origins(s, flagged) == ["user:A", "user:B"]
    assert responsible_origins(s, flagged, bounds=True) == ([], ["user:A", "user:B"])


def test_containment_audit_catches_a_skipped_entry():
    _, s, triggers = scenario()
    seeds = [triggers[1][0]]
    pre = len(s.tracker.ledger)
    c = contain(s, seeds)
    records = s.tracker.ledger.records()
    skip = next(r for r in records[pre:] if r["t"] == "contain")
    assert audit_containment([r for r in records if r is not skip], pre, seeds, c["run"])


def test_b3_parent_edge_replay_skips_displacement_and_is_measured():
    rng = random.Random(4)
    wrong = checked = 0
    for _ in range(200):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_external(script, k)
        cert = recover(s, [t for t, e in triggers if e], external_step_fn(s, k),
                       displaced=lambda *a: False)
        assert cert.stats["steps_replayed"]["retrieval"] == 0
        clean, _ = execute_external([x for x in script if not x[3]], k)
        wrong += state(s) != state(clean)
        checked += 1
    print(f"B3 diverged from the counterfactual on {wrong}/{checked} scripts")


def topic_step_fn(query, obs, hits):
    q = obs[0].text if obs else query
    if "=>" in q:
        core, adv = (x.strip() for x in q.split("=>", 1))
    else:
        core = q.strip()
        adv = core.split()[-1]
        if hits:
            hit_core, hit_adv = (x.strip() for x in hits[0].text.split("=>", 1))
            if hit_core.split()[1] == core.split()[1]:
                adv = hit_adv
    return [("mem", f"{core} => {adv}", {}),
            ("act", f"buy({adv})", {"reversible": False}),
            ("out", adv, {})]


def test_displacement_check_is_needed_for_exactness():
    rng = random.Random(0)
    diverged = 0
    for _ in range(400):
        script, k = random_script(rng), 1
        if not any(e for *_, e in script):
            continue
        clean, _ = execute_external([x for x in script if not x[3]], k, topic_step_fn)
        s, triggers = execute_external(script, k, topic_step_fn)
        seeds = [t for t, e in triggers if e]
        cert = recover(s, seeds, external_step_fn(s, k, topic_step_fn),
                       displaced=external_displaced(s, k))
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        b, tb = execute_external(script, k, topic_step_fn)
        recover(b, [t for t, e in tb if e], external_step_fn(b, k, topic_step_fn),
                displaced=lambda *a: False)
        if state(b) != state(clean):
            diverged += 1
            assert cert.stats["steps_replayed"]["retrieval"] > 0
    assert diverged > 0


@pytest.mark.parametrize("width", [2, 4, 8])
def test_parallel_replay_equals_sequential_and_counterfactual(width):
    rng = random.Random(5)
    conflicts = checked = 0
    for _ in range(150):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_external(script, k)
        cert = recover(s, [t for t, e in triggers if e], external_step_fn(s, k),
                       displaced=external_displaced(s, k), parallel=width)
        clean, _ = execute_external([x for x in script if not x[3]], k)
        assert state(s) == state(clean)
        assert verify_certificate(cert, s.tracker.ledger.records(), key=KEY) == []
        conflicts += cert.stats["parallel_conflicts"]
        checked += 1
    assert checked > 100
    assert conflicts > 0


def test_parallel_contain_then_repair_stays_safe():
    rng = random.Random(6)
    for _ in range(80):
        script, k = random_script(rng), rng.randint(1, 3)
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_external(script, k)
        seeds = [t for t, e in triggers if e]
        closure = {n for n in s.tracker.descendants(seeds) if s.tracker.nodes[n].kind == "mem"}
        contain(s, seeds)
        seen = []
        recover(s, seeds, watched(s, external_step_fn(s, k), closure, seen),
                displaced=external_displaced(s, k), parallel=4)
        assert not any(seen)
        clean, _ = execute_external([x for x in script if not x[3]], k)
        assert state(s) == state(clean)


def test_parallel_requires_agent_side_check():
    _, s, triggers = scenario()
    with pytest.raises(ValueError):
        recover(s, [triggers[1][0]], step_fn, parallel=2)


def test_audit_takes_products_from_the_log_not_the_replay_record():
    rng = random.Random(0)
    for _ in range(400):
        script, k = random_script(rng), 1
        if not any(e for *_, e in script):
            continue
        s, triggers = execute_external(script, k, topic_step_fn)
        cert = recover(s, [t for t, e in triggers if e], external_step_fn(s, k, topic_step_fn),
                       displaced=external_displaced(s, k))
        records = s.tracker.ledger.records()
        rep = next((r for r in records[cert.pre["size"]:]
                    if r["t"] == "replay" and r["cause"] == "retrieval" and r["old"]), None)
        if rep is None:
            continue
        hidden = rep["old"][0]
        tampered = [dict(r, old=r["old"][1:]) if r is rep else r for r in records
                    if not (r["t"] == "resolve" and r["id"] == hidden)]
        assert audit(records, cert.pre["size"], cert.seeds, cert.run) == []
        problems = audit(tampered, cert.pre["size"], cert.seeds, cert.run)
        assert any("missing from the replay record" in p for p in problems), problems
        return
    pytest.fail("no displaced step found")


def execute_without_writes(script, k, silent):
    s = MemoryStore(HashEmbedder())
    for i, (session, query, trust, _) in enumerate(script):
        ctx = s.begin_step(session)
        trigger = ctx.observe(query, f"user:{session}", trust, trigger=True)
        hits = ctx.retrieve(query, k)
        for kind, text, meta in step_fn(query, [trigger], hits):
            if not (kind == "mem" and i in silent):
                ctx.emit(kind, text, meta=meta)
    return s


def live_state(s):
    tr = s.tracker
    mem = [tr.nodes[i].text for i in sorted(s.active_ids(), key=lambda i: tr.nodes[i].order)]
    steps = [(st.session, [(tr.nodes[p].kind, tr.nodes[p].text) for p in st.products
                           if tr.nodes[p].status == "active"])
             for st in tr.steps if not st.dropped]
    return mem, steps


def test_deleting_entries_that_depend_on_each_other_equals_counterfactual():
    rng = random.Random(3)
    rerun_targets = checked = 0
    for _ in range(300):
        script, k = random_script(rng), rng.randint(1, 3)
        s, _ = execute(script, k)
        tr = s.tracker
        writers = [st.index for st in tr.steps if any(tr.nodes[p].kind == "mem" for p in st.products)]
        if len(writers) < 3:
            continue
        silent = set(rng.sample(writers, rng.randint(2, 3)))
        seeds = [p for i in silent for p in tr.steps[i].products if tr.nodes[p].kind == "mem"]
        cert = recover(s, seeds, step_fn, mode="delete")
        assert live_state(s) == live_state(execute_without_writes(script, k, silent))
        assert verify_certificate(cert, tr.ledger.records(), key=KEY) == []
        assert all(tr.nodes[x].text is None for x in seeds)
        rerun_targets += sum(1 for r in tr.ledger.records()[cert.pre["size"]:]
                             if r["t"] == "replay" and r.get("suppressed"))
        checked += 1
    assert checked > 100 and rerun_targets > 0
