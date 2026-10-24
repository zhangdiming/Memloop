import glob, json, itertools
from pathlib import Path
ROOT = Path.home() / "Memloop"
S = ["counterfactual", "counterfactual2", "recovered_all", "recovered_contain", "recovered_b3", "B1", "DP", "attack"]
agree = {}
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d)
    ev = json.load(open(d / "eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json"))
    ev.update(json.load(open(d / "eval_counterfactual2.json"))); ev.update(json.load(open(d / "eval_recovered_all.json")))
    R = {s: [r["reward"] == 1 for r in ev[s]["per_request"]] for s in S}
    for a, b in itertools.combinations(S, 2):
        agree.setdefault((a, b), []).extend(x == y for x, y in zip(R[a], R[b]))
for k, v in sorted(agree.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"{k[0]:18s} ~ {k[1]:18s} {sum(v)/len(v):.3f}")
