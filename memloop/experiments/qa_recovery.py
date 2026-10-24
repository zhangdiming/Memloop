from __future__ import annotations

import argparse, csv, json, os, pickle, random, sys, time
from pathlib import Path

from memloop import HashEmbedder, MemoryStore, audit_containment, contain, recover
from memloop.adapters.minja_qa import ATTACKER, MinjaQA, attacker_seeds
from memloop.llm import ChatLLM
from memloop.recovery import verify_certificate

KEY = b"memloop-minja-qa"
QA_DIR = Path(os.path.expanduser("~/Memloop/baselines/MINJA/QA"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def save(obj, p):
    pickle.dump(obj, open(p, "wb"))


def load(p):
    return pickle.load(open(p, "rb"))


def new_store():
    s = MemoryStore(HashEmbedder())
    s.tracker.ledger._key = KEY
    return s


def setup(args):
    random.seed(args.seed)
    rows = [r for r in csv.reader(open(QA_DIR / f"data/test/{args.subject}_test.csv")) if len(r) >= 6]
    contents = [dict(id=f"fixed_0_{i}", question=r[0].strip(),
                     options=f"A) {r[1].strip()}\nB) {r[2].strip()}\nC) {r[3].strip()}\nD) {r[4].strip()}",
                     groundtruth=r[5].strip()) for i, r in enumerate(rows)]
    victim_entry = json.load(open(QA_DIR / "victim.json"))[0]
    victim = victim_entry["victim"]
    notes = [list(n.values())[0] for n in victim_entry["note"]]
    templates = [c for c in contents if victim in c["question"].lower()]
    need = args.num_templates + args.num_test + args.num_eval
    if len(templates) < need:
        raise ValueError(f"{args.subject}: {len(templates)} '{victim}' questions, need {need}")
    tpl = random.sample(templates, args.num_templates)
    rest = [t for t in templates if t not in tpl]
    test = random.sample(rest, args.num_test)
    rest = [t for t in rest if t not in test]
    evals = random.sample(rest, args.num_eval)
    tpl_text = {t["question"] for t in templates}
    benign = [c for c in contents if c["question"] not in tpl_text]
    random.shuffle(benign)
    if args.overlap < 1:
        other = []
        for subj in args.other_subjects:
            other += [dict(id=f"{subj}_{i}", question=r[0].strip(),
                           options=f"A) {r[1].strip()}\nB) {r[2].strip()}\nC) {r[3].strip()}\nD) {r[4].strip()}",
                           groundtruth=r[5].strip())
                      for i, r in enumerate(csv.reader(open(QA_DIR / f"data/test/{subj}_test.csv")))
                      if len(r) >= 6 and victim not in r[0].lower()]
        random.shuffle(other)
        mixed = []
        while benign and other:
            mixed.append((benign if random.random() < args.overlap else other).pop())
        benign = mixed
    malicious_by_tpl = []
    for i, t in enumerate(tpl, start=1):
        group = [dict(t, id=f"fixed_{i}_{j}", question=f"{t['question']} {note}") for j, note in enumerate(notes)]
        group.append(dict(t, id=f"inject_{i}"))
        malicious_by_tpl.append(group)
    order = list(range(len(malicious_by_tpl)))
    random.shuffle(order)
    malicious = [q for i in order for q in malicious_by_tpl[i]]
    test = [dict(t, id=f"test_{i}") for i, t in enumerate(test, start=1)]
    evals = [dict(t, id=f"eval_{i}") for i, t in enumerate(evals, start=1)]
    return benign, malicious, test, evals


_EMBEDDER = None


def harness(args, store, propagate):
    global _EMBEDDER
    llm = ChatLLM(args.model, temperature=args.temperature, seed=args.seed, max_tokens=1500)
    demo = (QA_DIR / "initial_demo.txt").read_text()
    if args.retriever != "lev" and _EMBEDDER is None:
        from memloop import STEmbedder
        _EMBEDDER = STEmbedder(args.embed_model, device="cpu")
    return MinjaQA(llm, store, demo, n_shots=args.n_shots, propagate=propagate,
                   retriever=args.retriever, embedder=_EMBEDDER, evolve=args.evolve)


def phase_attack(args, out):
    benign, malicious, test, evals = setup(args)
    store = new_store()
    h = harness(args, store, args.propagate)
    script = []

    def run(q, kind, owner):
        st = h.run_step(q, kind, owner)
        o = json.loads(store.tracker.nodes[st.products[-1]].text)
        script.append(dict(step=st.index, session=q["id"], kind=kind, owner=owner, q=q, **o,
                           stored=any(store.tracker.nodes[p].kind == "mem" for p in st.products)))
        log(f"step {st.index} {kind:9s} {q['id']:12s} answer={o['answer']} gt={q['groundtruth']} "
            f"correct={o['correct']} injected={o['hit_target']} stored={script[-1]['stored']} "
            f"retrieved={len(st.retrieved)}")
        return o

    order = [1] * len(malicious) + [0] * args.num_benign
    random.shuffle(order)
    if args.attack_window:
        n = len(order)
        lo, hi = int(args.attack_window[0] * n), int(args.attack_window[1] * n)
        if hi - lo < len(malicious):
            raise ValueError(f"attack window holds {hi - lo} slots, need {len(malicious)}")
        slots = set(random.sample(range(lo, hi), len(malicious)))
        order = [int(i in slots) for i in range(n)]
    mi = bi = 0
    for slot in order:
        if slot == 1:
            q = malicious[mi]; mi += 1
            for _ in range(3):
                if run(q, "malicious", ATTACKER)["hit_target"]:
                    break
        else:
            for _ in range(args.max_benign_retries):
                if bi >= len(benign):
                    break
                q = benign[bi]; bi += 1
                if run(q, "benign", f"benign-{bi}")["correct"]:
                    break
    for i, q in enumerate(test, start=1):
        run(q, "test", f"victim-{i}")
    json.dump(dict(script=script, evals=evals, args=vars(args)), open(out / "script.json", "w"), indent=1)
    save(dict(store=store), out / "attack.pkl")
    tpl_inj = [e for e in script if e["session"].startswith("inject_")]
    isr = len({e["session"] for e in tpl_inj if e["hit_target"]}) / args.num_templates
    asr = sum(e["hit_target"] for e in script if e["kind"] == "test") / len(test)
    log(f"attack done: {len(script)} steps, ISR {isr:.2f}, test ASR {asr:.2f}")


def phase_recover(args, out):
    store = load(out / "attack.pkl")["store"]
    h = harness(args, store, args.propagate)
    seeds = attacker_seeds(store)
    if args.seeding != "oracle":
        from memloop.attribution import responsible_origins, seeds_for_entries, seeds_for_origins
        flags = json.load(open(out / "detector_flags.json"))
        flagged = [n for n, v in flags.items() if v["flag"]]
        seeds = (seeds_for_entries(store, flagged) if args.seeding == "entry"
                 else seeds_for_origins(store, responsible_origins(store, flagged)))
        log(f"seeding={args.seeding}: {len(flagged)} flagged -> {len(seeds)} seeds")
    tag = "recovered_contain" + ("" if args.seeding == "oracle" else f"_{args.seeding}") + (
        "_b3" if args.variant == "b3" else "")
    if not seeds:
        save(dict(store=store), out / f"{tag}.pkl")
        log("no seeds: nothing to recover")
        return
    pre = len(store.tracker.ledger)
    c = contain(store, seeds)
    cprob = audit_containment(store.tracker.ledger.records(), pre, seeds, c["run"])
    t0 = time.time()
    displaced = (lambda *a: False) if args.variant == "b3" else h.displaced
    cert = recover(store, seeds, h.step_fn, displaced=displaced, parallel=args.parallel)
    stats = {**cert.stats, "steps_replayed": dict(cert.stats["steps_replayed"]),
             "seconds": time.time() - t0, "contain_seconds": c["seconds"], "contained": len(c["contained"]),
             "certificate_problems": verify_certificate(cert, store.tracker.ledger.records(), key=KEY),
             "containment_problems": cprob,
             "replay_timeline": [(round(r["t"], 2), r["step"], r["cause"]) for r in cert.replayed]}
    save(dict(store=store), out / f"{tag}.pkl")
    json.dump(stats, open(out / f"{tag}_stats.json", "w"), indent=1)
    log(f"{tag}: { {k: v for k, v in stats.items() if k != 'replay_timeline'} }")


def phase_counterfactual(args, out):
    sc = json.load(open(out / "script.json"))
    store = new_store()
    h = harness(args, store, args.propagate)
    for e in sc["script"]:
        if e["owner"] != ATTACKER:
            h.run_step(e["q"], e["kind"], e["owner"])
    save(dict(store=store), out / f"{args.cf_tag}.pkl")
    log(f"{args.cf_tag} done")


def phase_baselines(args, out):
    for tag in ("B1", "B2"):
        store = load(out / "attack.pkl")["store"]
        tr, seeds = store.tracker, attacker_seeds(store)
        if tag == "B1":
            doomed = [n for n in tr.nodes.values() if n.kind == "mem" and tr.steps[n.step].session == ATTACKER]
        else:
            doomed = [tr.nodes[n] for n in tr.descendants(seeds) if tr.nodes[n].kind == "mem"]
        drop(store, doomed)
        save(dict(store=store), out / f"{tag}.pkl")
        log(f"{tag}: removed {len(doomed)}, {len(store.active_ids())} left")


def drop(store, nodes):
    gone = {n.id for n in nodes}
    for n in sorted(nodes, key=lambda n: n.order):
        if store.has(n.id):
            store.remove(n.id)
        old = n.meta.get("supersedes")
        if old and old not in gone and store.retired_by(old) == n.id:
            store.restore(old)


def phase_eval(args, out):
    sc = json.load(open(out / "script.json"))
    res = {}
    for tag in args.eval_on:
        p = out / f"{tag}.pkl"
        if not p.exists():
            continue
        store = load(p)["store"]
        h = harness(args, store, propagate=False)
        end = len(store.tracker.steps)
        hits = sum(h.episode(q, "eval", end).injected for q in sc["evals"])
        res[tag] = dict(asr=hits / len(sc["evals"]), hits=hits, n=len(sc["evals"]),
                        memory=sum(store.tracker.nodes[i].kind == "mem" for i in store.active_ids()))
        log(f"eval {tag}: {res[tag]}")
    json.dump(res, open(out / f"eval_{'_'.join(args.eval_on)}.json", "w"), indent=1)


def rows(store):
    tr, out = store.tracker, []
    for st in tr.steps:
        if st.dropped or st.session == ATTACKER:
            continue
        nodes = [tr.nodes[p] for p in st.products]
        prods = {n.kind: n for n in nodes if not n.meta.get("supersedes")}
        o = json.loads(prods["out"].text)
        evo = [(n.meta["record"]["id"], n.text) for n in nodes if n.meta.get("supersedes")]
        out.append(dict(session=tr.nodes[st.trigger].meta["q"]["id"], kind=tr.nodes[st.trigger].meta["kind"],
                        step=st.index, mem=prods["mem"].meta["record"]["thought"] if "mem" in prods else None,
                        **({"evo": evo} if evo else {}), **o))
    return out


def live_memory(store):
    return {store.tracker.nodes[i].text for i in store.active_ids()
            if store.tracker.nodes[i].kind == "mem"}


def phase_compare(args, out):
    cf_store = load(out / "counterfactual.pkl")["store"]
    cf, cf_mem = rows(cf_store), live_memory(cf_store)
    sc = json.load(open(out / "script.json"))
    tau = min(e["step"] for e in sc["script"] if e["owner"] == ATTACKER)
    steps = [e["step"] for e in sc["script"] if e["owner"] != ATTACKER]
    rep = {}
    def summ(r):
        tests = [x for x in r if x["kind"] == "test"]
        ben = [x for x in r if x["kind"] == "benign"]
        return dict(test_asr=sum(x["hit_target"] for x in tests) / max(1, len(tests)),
                    benign_acc=sum(x["correct"] for x in ben) / max(1, len(ben)),
                    stored=sum(x["mem"] is not None for x in r),
                    rewrites=sum(len(x.get("evo", ())) for x in r))
    for tag in ("attack", "recovered_contain", "recovered_contain_b3", "counterfactual2"):
        p = out / f"{tag}.pkl"
        if not p.exists():
            continue
        st = load(p)["store"]
        r, mem = rows(st), live_memory(st)
        assert [x["session"] for x in r] == [x["session"] for x in cf], f"{tag}: scripts diverge"
        n = len(cf)
        same = [{**a, "step": 0} == {**b, "step": 0} for a, b in zip(r, cf)]
        late = [s for s, t in zip(same, steps) if t > tau]
        rep[tag] = dict(episodes=n, exact=sum(same) / n, after_tau=len(late),
                        differ_after_tau=len(late) - sum(late),
                        memory=len(mem), memory_cf=len(cf_mem),
                        memory_equal=mem == cf_mem,
                        memory_jaccard=len(mem & cf_mem) / max(1, len(mem | cf_mem)),
                        outcome=sum((a["answer"], a["mem"] is not None) == (b["answer"], b["mem"] is not None)
                                    for a, b in zip(r, cf)) / n, **summ(r))
    rep["counterfactual"] = summ(cf)
    json.dump(rep, open(out / "compare.json", "w"), indent=1)
    for k, v in rep.items():
        log(f"compare {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["attack", "recover", "counterfactual", "baselines", "eval", "compare"])
    ap.add_argument("--subject", default="nutrition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="qwen3-14b")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-templates", type=int, default=10)
    ap.add_argument("--num-test", type=int, default=10)
    ap.add_argument("--num-eval", type=int, default=10)
    ap.add_argument("--num-benign", type=int, default=30)
    ap.add_argument("--n-shots", type=int, default=3)
    ap.add_argument("--max-benign-retries", type=int, default=10)
    ap.add_argument("--propagate", action="store_true")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--seeding", choices=["oracle", "entry", "origin"], default="oracle")
    ap.add_argument("--variant", choices=["contain", "b3"], default="contain")
    ap.add_argument("--retriever", choices=["lev", "dense-q", "dense-qt", "amem"], default="lev")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--other-subjects", nargs="+",
                    default=["high_school_geography", "world_religions", "marketing", "astronomy"])
    ap.add_argument("--attack-window", type=float, nargs=2, default=None)
    ap.add_argument("--cf-tag", default="counterfactual")
    ap.add_argument("--eval-on", nargs="+", default=["attack", "B1", "B2", "recovered_contain", "counterfactual"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    globals()[f"phase_{args.phase}"](args, out)


if __name__ == "__main__":
    main()
