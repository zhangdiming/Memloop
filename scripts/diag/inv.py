import glob, json, pickle, sys
from collections import Counter
from pathlib import Path
ROOT = Path.home() / "Memloop"
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import ATTACKER
S = ["counterfactual", "recovered_all", "recovered_contain", "recovered_b3"]
tot = Counter(); ex = {}
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d)
    for s in S:
        st = pickle.load(open(d / f"{s}.pkl", "rb")); store, calls = st["store"], st["calls"]; tr = store.tracker
        live = set(store.active_ids())
        for step in tr.steps:
            if step.dropped or step.session == ATTACKER: continue
            cs = [c for c in calls.get(step.index, []) if c["mode"] == "plan"]
            if not cs: continue
            exp = [n for n in store.live_before(step.index) if n.kind == "mem"]
            tot[(s, "steps")] += 1
            if cs[0]["n_mem"] != len(exp):
                tot[(s, "n_mem_mismatch")] += 1; tot[(s, "n_mem_diff_sum")] += cs[0]["n_mem"] - len(exp)
                ex.setdefault(s, (d.name, d.parent.name, step.index, cs[0]["n_mem"], len(exp)))
            for c in calls[step.index]:
                for h in c["hits"]:
                    n = tr.nodes.get(h)
                    if n is None: tot[(s, "hit_unknown")] += 1
                    elif n.step >= step.index: tot[(s, "hit_future")] += 1
                    elif h not in live and n.status != "superseded": tot[(s, "hit_now_dead:" + n.status)] += 1
for k in sorted(tot): print(k, tot[k])
print(ex)
