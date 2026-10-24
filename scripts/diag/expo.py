import glob, json, pickle, sys
from pathlib import Path
ROOT = Path.home() / "Memloop"
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import episodes, ATTACKER, MARK
S = ["attack", "counterfactual", "counterfactual2", "recovered_all"]
tot = {}
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d); sc = json.load(open(d / "script.json"))["script"]
    tau = next(i for i, e in enumerate(sc) if e["owner"] == ATTACKER)
    A = pickle.load(open(d / "attack.pkl", "rb")); tr = A["store"].tracker
    bad = {i for i, n in tr.nodes.items() if n.kind == "mem" and (tr.steps[n.step].session == ATTACKER or MARK in n.text.lower())}
    R = {s: {r["session"]: r["reward"] == 1 for r in episodes((A if s == "attack" else pickle.load(open(d / f"{s}.pkl", "rb")))["store"])} for s in S}
    for e in sc[tau:]:
        if e["owner"] == ATTACKER: continue
        hits = {h for c in A["calls"].get(e["step"], []) for h in c["hits"]}
        g = (e["kind"], "exposed" if hits & bad else "unexposed")
        for s in S: tot.setdefault(g + (s,), []).append(R[s][e["session"]])
for k in sorted(tot):
    v = tot[k]; print(f"{k[0]:7s} {k[1]:10s} {k[2]:16s} n={len(v):4d} success {sum(v)/len(v):.3f}")
