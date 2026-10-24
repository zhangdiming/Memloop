import glob, json, pickle, re, sys
from pathlib import Path
import numpy as np
ROOT = Path.home() / "Memloop"
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import ATTACKER
S = ["attack", "counterfactual", "counterfactual2", "recovered_all", "recovered_contain", "B1", "DP"]
rows = []
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d); pair = json.load(open(d / "script.json"))["pair"]; v = pair["victim"]
    ev = json.load(open(d / "eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json"))
    ev.update(json.load(open(d / "eval_counterfactual2.json"))); ev.update(json.load(open(d / "eval_recovered_all.json")))
    row = {"run": f"{d.parent.name}/{d.name}"}
    for s in S:
        st = pickle.load(open(d / f"{s}.pkl", "rb"))["store"]; tr = st.tracker
        mem = [tr.nodes[i] for i in st.active_ids() if tr.nodes[i].kind == "mem"]
        sess = lambda n: tr.steps[n.step].session
        row[s] = dict(vict=sum(bool(re.search(rf"\b{v}\b", json.loads(n.text)["Task"])) and sess(n) != ATTACKER for n in mem),
                      test=sum(sess(n).startswith("test_") for n in mem), n=len(mem),
                      vs=ev["recovered_all" if s == "recovered_all" else s]["victim_success"] if s != "recovered_all" or True else 0)
    rows.append(row)
for s in S:
    print(f"{s:18s} victim-task records {np.mean([r[s]['vict'] for r in rows]):5.2f}  of which in-script victim tests {np.mean([r[s]['test'] for r in rows]):5.2f}  records {np.mean([r[s]['n'] for r in rows]):5.1f}  eval {np.mean([r[s]['vs'] for r in rows]):.3f}")
cf = lambda r, k: (r["counterfactual"][k] + r["counterfactual2"][k]) / 2
for s in ("recovered_all", "recovered_contain", "B1", "DP"):
    x = np.array([r[s]["vict"] - cf(r, "vict") for r in rows]); y = np.array([r[s]["vs"] - cf(r, "vs") for r in rows])
    b, a = np.polyfit(x, y, 1); res = y - a - b * x
    print(f"{s:18s} d_victim_records {x.mean():+.2f}±{1.96*x.std(ddof=1)/len(x)**.5:.2f}  eval gap {y.mean():+.3f}  r={np.corrcoef(x,y)[0,1]:+.2f} slope {b:+.3f}/record  intercept {a:+.3f}±{1.96*res.std(ddof=2)/len(y)**.5:.3f}")
json.dump(rows, open(ROOT / "paper/data/alfworld_gap_memory.json", "w"), indent=1)
