import glob, json, os, pickle, random, sys, zlib
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))
sys.path.insert(0, os.path.dirname(__file__))
from memloop.attribution import responsible_origins, seeds_for_origins
from memloop.labels import Label
from attribution_adaptive import ATT, BASE, poisoned, relabel


def fold_benign(store, K):
    if K is None:
        return
    tr, acct = store.tracker, {}
    for st in tr.steps:
        if st.session == ATT or st.trigger is None:
            continue
        t = tr.nodes[st.trigger]
        o = next(iter(t.label.origins))
        acct.setdefault(o, len(acct) % K)
        t.label = Label(t.label.trust, frozenset([f"user:user-{acct[o]}"]))


def flags_llm(d, store, rng):
    f = json.load(open(f"{d}/detector_flags.json"))
    return {n for n, v in f.items() if v["flag"] and n in store.tracker.nodes}


def flags_sim(recall, fp):
    def g(d, store, rng):
        tr = store.tracker
        live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
        att = {i for i in live if tr.steps[tr.nodes[i].step].session == ATT}
        pois = sorted(att | {i for i in live if i not in att and poisoned(tr.nodes[i].meta["record"])})
        out = set(rng.sample(pois, round(recall * len(pois)))) if pois else set()
        return out | {i for i in live if i not in pois and rng.random() < fp}
    return g


def run(K, k, detector, name, reps):
    agg = defaultdict(list)
    for d in sorted(glob.glob(f"{BASE}/*/seed*")):
        for rep in range(reps):
            rng = random.Random(zlib.crc32(repr((d, rep, name)).encode()))
            store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
            tr = store.tracker
            relabel(store, k)
            fold_benign(store, K)
            flagged = detector(d, store, rng)
            lo, hi = responsible_origins(store, flagged, bounds=True)
            seeds = set(seeds_for_origins(store, hi))
            att_steps = {st.trigger for st in tr.steps if st.session == ATT}
            ben_steps = {st.trigger for st in tr.steps if st.session != ATT and st.trigger}
            agg["att_seeded"].append(len(seeds & att_steps) / max(1, len(att_steps)))
            agg["innocent_named"].append(sum(not o.startswith("user:attacker") for o in hi))
            agg["benign_seeded"].append(len(seeds & ben_steps) / max(1, len(ben_steps)))
            agg["cycle"].append(lo != hi)
            agg["any_innocent"].append(any(not o.startswith("user:attacker") for o in hi))
    n = len(agg["att_seeded"])
    return {m: sum(v) / n for m, v in agg.items()} | {"n": n}


if __name__ == "__main__":
    dets = [("llm", flags_llm, 1)] + [(f"sim r={r} fp={f}", flags_sim(r, f), 3)
                                      for r in (1.0, 0.5) for f in (0.0, 0.05, 0.10)]
    rows = []
    print(f"{'detector':16s} {'K':>4s} {'k':>2s} | {'att seeded':>10s} {'innoc.named':>11s} "
          f"{'runs w/ innoc.':>14s} {'benign seeded':>13s} {'cycle':>6s}")
    for name, det, reps in dets:
        for k in (1, 2):
            for K in (None, 20, 5, 2):
                r = run(K, k, det, name, reps)
                rows.append(dict(detector=name, K=K, k=k, **r))
                print(f"{name:16s} {str(K):>4s} {k:2d} | {r['att_seeded']:10.0%} {r['innocent_named']:11.2f} "
                      f"{r['any_innocent']:14.0%} {r['benign_seeded']:13.1%} {r['cycle']:6.0%}", flush=True)
    out = os.path.expanduser("~/Memloop/results/attribution_recurring.json")
    json.dump(rows, open(out, "w"), indent=1)
