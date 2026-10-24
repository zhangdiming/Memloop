import glob, json, os, pickle, random, sys, zlib
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))
from memloop.attribution import minimal_flagged, responsible_origins, seeds_for_origins
from memloop.labels import Label

BASE = os.path.expanduser("~/Memloop/results/minja_recovery/qwen3-14b-main")


def poisoned(rec):
    return any("out of stock" in a.lower() for a in rec["Actions"])


def load(d, sybil):
    store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
    tr = store.tracker
    if sybil:
        for st in tr.steps:
            if st.session == "attacker":
                t = tr.nodes[st.trigger]
                t.label = Label(t.label.trust, frozenset([f"user:attacker-{st.index}"]))
    live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
    att = {i for i in live if tr.steps[tr.nodes[i].step].session == "attacker"}
    cont = {i for i in live if i not in att and poisoned(tr.nodes[i].meta["record"])}
    benign = set(live) - att - cont
    return store, att, cont, benign


def run(recall, fp, sybil, reps=3):
    agg = defaultdict(list)
    for d in sorted(glob.glob(f"{BASE}/*/seed*")):
        for rep in range(reps):
            rng = random.Random(zlib.crc32(repr((d, rep, recall, fp)).encode()))
            store, att, cont, benign = load(d, sybil)
            tr = store.tracker
            pois = sorted(att | cont)
            flagged = set(rng.sample(pois, round(recall * len(pois)))) if pois else set()
            flagged |= {b for b in benign if rng.random() < fp}
            origins = responsible_origins(store, flagged)
            is_att = lambda o: o.startswith("user:attacker")
            att_origins = {next(iter(tr.nodes[st.trigger].label.origins))
                           for st in tr.steps if st.session == "attacker"}
            seeds = set(seeds_for_origins(store, origins))
            att_steps = {st.trigger for st in tr.steps if st.session == "attacker"}
            agg["runs"].append(1)
            agg["attacker_named"].append(any(is_att(o) for o in origins))
            agg["innocent_named"].append(sum(not is_att(o) for o in origins))
            agg["attacker_sessions_seeded"].append(len(seeds & att_steps) / max(1, len(att_steps)))
            agg["innocent_sessions_seeded"].append(len(seeds - att_steps))
            agg["entry_level_missed_attacker_entries"].append(len(att - flagged) / max(1, len(att)))
            mins = minimal_flagged(store, flagged)
            agg["minimal_are_attacker"].append(len(mins & att) / max(1, len(mins)))
            agg["n_attacker_origins"].append(len(att_origins))
    n = len(agg["runs"])
    return {k: sum(v) / n for k, v in agg.items() if k != "runs"} | {"n": n}


if __name__ == "__main__":
    print(f"{'sybil':5s} {'recall':>6s} {'fp':>4s} | {'attacker named':>14s} {'innocents named':>15s} "
          f"{'att. sessions seeded':>20s} {'innocent sessions seeded':>24s} | {'entry-level: att. entries missed':>32s}")
    for sybil in (False, True):
        for recall in (1.0, 0.75, 0.5, 0.25):
            for fp in (0.0, 0.05, 0.10):
                r = run(recall, fp, sybil)
                print(f"{str(sybil):5s} {recall:6.2f} {fp:4.2f} | {r['attacker_named']:14.0%} {r['innocent_named']:15.2f} "
                      f"{r['attacker_sessions_seeded']:20.0%} {r['innocent_sessions_seeded']:24.2f} | "
                      f"{r['entry_level_missed_attacker_entries']:32.0%} {r['minimal_are_attacker']:16.0%}")
