import glob, json, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path.home() / "Memloop"
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import ATTACKER, MARK
S = ["attack", "counterfactual", "counterfactual2", "recovered_all", "recovered_contain", "B1", "DP"]
acc = defaultdict(list)
for d in sorted(glob.glob(str(ROOT / "results/alfworld/qwen3-14b-main/*/seed*"))):
    d = Path(d); sc = json.load(open(d / "script.json")); pair = sc["pair"]; v, t = pair["victim"], pair["target"]
    tau_sess = {e["session"] for e in sc["script"][:next(i for i, e in enumerate(sc["script"]) if e["owner"] == ATTACKER)]}
    for s in S:
        st = pickle.load(open(d / f"{s}.pkl", "rb"))["store"]; tr = st.tracker
        for i in st.active_ids():
            n = tr.nodes[i]
            if n.kind != "mem" or tr.steps[n.step].session == ATTACKER: continue
            r = json.loads(n.text); seg = "pre" if tr.steps[n.step].session in tau_sess else "post"
            acts = " ".join(r["Actions"]).lower()
            k = (s, seg)
            acc[k + ("n",)].append(1)
            acc[k + ("double_plan",)].append(r["Plan"].count("To solve the task") > 1)
            acc[k + ("take_target",)].append(bool(re.search(rf"take {t}\b", acts)))
            acc[k + ("mark",)].append(MARK in n.text.lower())
            acc[k + ("len_actions",)].append(len(r["Actions"]))
            acc[k + ("victim_task",)].append(bool(re.search(rf"\b{v}\b", r["Task"])))
            acc[k + ("ok_status",)].append(n.status == "active")
for seg in ("pre", "post"):
    for s in S:
        f = lambda m: np.mean(acc[(s, seg, m)]) if acc[(s, seg, m)] else float("nan")
        print(f"{seg:4s} {s:18s} records/run {len(acc[(s, seg, 'n')])/27:5.1f} double_plan {f('double_plan'):.3f} take_target {f('take_target'):.3f} mark {f('mark'):.3f} len {f('len_actions'):5.1f} victim {f('victim_task'):.3f}")
