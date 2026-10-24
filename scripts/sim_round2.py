import json, os, pickle, random, sys, zlib
sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))
from memloop.attribution import responsible_origins, seeds_for_origins
d = os.path.expanduser("~/Memloop/results/minja_recovery/qwen3-14b-main/camera/seed2")
def poisoned(rec): return any("out of stock" in a.lower() for a in rec["Actions"])
for t, recall in (("r075_f05_d0", .75), ("r050_f05_d0", .5)):
    store = pickle.load(open(f"{d}/recovered_contain_sim_{t}.pkl", "rb"))["store"]
    tr = store.tracker
    live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
    att = {i for i in live if tr.steps[tr.nodes[i].step].session == "attacker"}
    cont = {i for i in live if i not in att and poisoned(tr.nodes[i].meta["record"])}
    benign = set(live) - att - cont
    pois = sorted(att | cont)
    rng = random.Random(zlib.crc32(repr((d, 0, recall, 0.05, 2)).encode()))
    flagged = set(rng.sample(pois, round(recall * len(pois)))) if pois else set()
    flagged |= {b for b in benign if rng.random() < 0.05}
    json.dump({i: dict(flag=i in flagged) for i in live}, open(f"{d}/detector_flags_sim_{t}_r2.json", "w"))
    o = responsible_origins(store, flagged)
    seeds = set(seeds_for_origins(store, o))
    att_steps = {st.trigger for st in tr.steps if st.session == "attacker"}
    print(t, "live", len(live), "att", len(att), "cont", len(cont), "flagged", len(flagged),
          "origins", sorted(o), "seeds", len(seeds), "att_steps", len(att_steps), "covers", att_steps <= seeds, "extra", len(seeds - att_steps))
