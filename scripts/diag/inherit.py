import glob, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path.home() / "Memloop"
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import episodes, ATTACKER, MARK
S = ["attack", "counterfactual", "counterfactual2", "recovered_all"]
X, XS, Y, YC = [], [], [], []
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d); sc = json.load(open(d / "script.json"))["script"]
    tau = next(i for i, e in enumerate(sc) if e["owner"] == ATTACKER)
    P = {s: pickle.load(open(d / f"{s}.pkl", "rb")) for s in S}
    tr = P["attack"]["store"].tracker
    bad = {i for i, n in tr.nodes.items() if n.kind == "mem" and (tr.steps[n.step].session == ATTACKER or MARK in n.text.lower())}
    R = {s: {r["session"]: (r["reward"] == 1, r["mem"] is not None) for r in episodes(P[s]["store"])} for s in S}
    un = [e["session"] for e in sc[tau:] if e["owner"] != ATTACKER and
          not ({h for c in P["attack"]["calls"].get(e["step"], []) for h in c["hits"]} & bad)]
    m = lambda s, j=0: np.mean([R[s][x][j] for x in un])
    X.append(m("attack") - (m("counterfactual") + m("counterfactual2")) / 2)
    XS.append(sum(R["attack"][x][1] for x in un) - (sum(R["counterfactual"][x][1] for x in un) + sum(R["counterfactual2"][x][1] for x in un)) / 2)
    ev = json.load(open(d / "eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json"))
    ev.update(json.load(open(d / "eval_counterfactual2.json"))); ev.update(json.load(open(d / "eval_recovered_all.json")))
    cf = (ev["counterfactual"]["victim_success"] + ev["counterfactual2"]["victim_success"]) / 2
    Y.append(ev["recovered_all"]["victim_success"] - cf); YC.append(ev["recovered_contain"]["victim_success"] - cf)
X, XS, Y, YC = map(np.array, (X, XS, Y, YC))
print("x (attack - fresh, unexposed post-tau success) mean %.3f sd %.3f" % (X.mean(), X.std(ddof=1)))
print("xs (stored records, same) mean %.2f" % XS.mean())
for name, y in (("all", Y), ("contain", YC)):
    for xn, x in (("success", X), ("stored", XS)):
        b, a = np.polyfit(x, y, 1); r = np.corrcoef(x, y)[0, 1]
        res = y - (a + b * x); se = res.std(ddof=2) / len(y) ** .5
        print(f"{name:8s} vs {xn:8s}: r={r:+.2f} slope={b:+.3f} intercept={a:+.3f}±{1.96*se:.3f} (raw gap {y.mean():+.3f})")
