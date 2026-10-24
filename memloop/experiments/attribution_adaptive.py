import glob, json, os, pickle, random, sys, zlib
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))
from memloop.attribution import responsible_origins, seeds_for_origins
from memloop.labels import Label

BASE = os.path.expanduser("~/Memloop/results/minja_recovery/qwen3-14b-main")
ATT = "attacker"


def poisoned(rec):
    return any("out of stock" in a.lower() for a in rec["Actions"])


def relabel(store, k):
    tr = store.tracker
    sessions = []
    for st in tr.steps:
        if st.session == ATT:
            s = tr.nodes[st.trigger].meta["session"]
            if s not in sessions:
                sessions.append(s)
    acct = {s: (i if k is None else i % k) for i, s in enumerate(sessions)}
    for st in tr.steps:
        if st.session == ATT:
            t = tr.nodes[st.trigger]
            t.label = Label(t.label.trust, frozenset([f"user:attacker-{acct[t.meta['session']]}"]))
    return len(sessions)


def one(d, k, flags_of):
    store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
    tr = store.tracker
    n_sess = relabel(store, k)
    flagged = flags_of(d, store)
    origins = responsible_origins(store, flagged)
    seeds = set(seeds_for_origins(store, origins))
    att_steps = {st.trigger for st in tr.steps if st.session == ATT}
    live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
    att_mem = {i for i in live if tr.steps[tr.nodes[i].step].session == ATT}
    seeded_steps = {tr.nodes[x].step for x in seeds}
    by_origin = {i for i in att_mem if tr.nodes[i].step in seeded_steps}
    fl = set(flagged)
    return dict(left_origin=len(att_mem - by_origin) / max(1, len(att_mem)),
                left_entry=len(att_mem - fl) / max(1, len(att_mem)),
                left_union=len(att_mem - by_origin - fl) / max(1, len(att_mem)),
                clean_union=not (att_mem - by_origin - fl),
                sessions=n_sess,
                accounts=n_sess if k is None else min(k, n_sess),
                named=sum(o.startswith("user:attacker") for o in origins),
                innocent=sum(not o.startswith("user:attacker") for o in origins),
                covered=len(seeds & att_steps) / max(1, len(att_steps)),
                full=att_steps <= seeds,
                entry_missed=len(att_mem - set(flagged)) / max(1, len(att_mem)))


def llm_flags(d, store):
    f = json.load(open(f"{d}/detector_flags.json"))
    return [nid for nid, v in f.items() if v["flag"]]


def sim_flags(recall, fp):
    def flags(d, store):
        tr = store.tracker
        rng = random.Random(zlib.crc32(repr((d, recall, fp)).encode()))
        live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
        att = {i for i in live if tr.steps[tr.nodes[i].step].session == ATT}
        cont = {i for i in live if i not in att and poisoned(tr.nodes[i].meta["record"])}
        pois = sorted(att | cont)
        out = set(rng.sample(pois, round(recall * len(pois)))) if pois else set()
        return sorted(out | {b for b in live if b not in att | cont and rng.random() < fp})
    return flags


if __name__ == "__main__":
    runs = sorted(glob.glob(f"{BASE}/*/seed*"))
    detectors = {"llm": llm_flags, "sim r=1.0 fp=0.05": sim_flags(1.0, 0.05),
                 "sim r=0.5 fp=0.05": sim_flags(0.5, 0.05)}
    report = {}
    for name, fl in detectors.items():
        for k in (1, 2, 3, 5, None):
            rows = [one(d, k, fl) for d in runs]
            agg = defaultdict(float)
            for r in rows:
                for key, v in r.items():
                    agg[key] += float(v) / len(rows)
            key = f"{name} | k={'per-session' if k is None else k}"
            report[key] = dict(agg) | {"runs": len(rows),
                                       "any_named": sum(r["named"] > 0 for r in rows),
                                       "full_cover": sum(r["full"] for r in rows),
                                       "clean_union": sum(r["clean_union"] for r in rows)}
            print(f"{key:32s} named-any {report[key]['any_named']:2d}/{len(rows)}  "
                  f"full-cover {report[key]['full_cover']:2d}/{len(rows)}  "
                  f"att-sessions-seeded {agg['covered']:.0%}  innocents {agg['innocent']:.2f}  "
                  f"left: origin {agg['left_origin']:.0%} entry {agg['left_entry']:.0%} "
                  f"union {agg['left_union']:.0%} (clean {report[key]['clean_union']}/{len(rows)})", flush=True)
    json.dump(report, open(f"{BASE}/attribution_adaptive.json", "w"), indent=1)
