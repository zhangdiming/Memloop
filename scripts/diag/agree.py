import glob, json, pickle, sys
from pathlib import Path
ROOT = Path.home() / "Memloop"
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import episodes, ATTACKER
S = ["attack", "counterfactual", "counterfactual2", "recovered_all", "recovered_contain"]
tot = {}
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d); sc = json.load(open(d / "script.json"))["script"]
    tau = next(i for i, e in enumerate(sc) if e["owner"] == ATTACKER)
    post = {e["session"]: e["kind"] for e in sc[tau:] if e["owner"] != ATTACKER}
    R = {s: {r["session"]: r["reward"] == 1 for r in episodes(pickle.load(open(d / f"{s}.pkl", "rb"))["store"])} for s in S}
    for sess, kind in post.items():
        a = R["attack"][sess]
        for s in S[1:]:
            k = (kind, "attack_" + ("ok" if a else "fail"), s)
            tot.setdefault(k, []).append(R[s][sess])
        for x, y in [("counterfactual2", "counterfactual"), ("recovered_all", "attack"), ("counterfactual", "attack"),
                     ("counterfactual2", "attack"), ("recovered_all", "counterfactual")]:
            tot.setdefault((kind, "agree", f"{x}~{y}"), []).append(R[x][sess] == R[y][sess])
for k in sorted(tot):
    v = tot[k]; print(f"{k[0]:8s} {k[1]:12s} {k[2]:34s} n={len(v):4d} {sum(v)/len(v):.3f}")
