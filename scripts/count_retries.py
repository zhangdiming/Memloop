import json, pickle, sys
from pathlib import Path
sys.path.insert(0, "memloop")
ATTACKER = "attacker"


def episodes(store):
    tr, rows = store.tracker, []
    for st in tr.steps:
        if st.dropped or st.session == ATTACKER:
            continue
        outc = next(json.loads(tr.nodes[p].text) for p in st.products if tr.nodes[p].kind == "out")
        rows.append(dict(session=tr.nodes[st.trigger].meta["session"], reward=outc["reward"]))
    return rows
MAIN = Path("results/minja_recovery/qwen3-14b-main")
tot = dict(benign=0, after_fail=0, fail_att_ok_cf=0, fail_cf_ok_cf2=0, fail_cf=0)
rows = []
for d in sorted(MAIN.glob("*/seed*")):
    sc = json.load(open(d / "script.json"))["script"]
    ben = [s for s in sc if s["kind"] == "benign"]
    cf = {r["session"]: r for r in episodes(pickle.load(open(d / "counterfactual.pkl", "rb"))["store"])}
    cf2p = d / "counterfactual2.pkl"
    cf2 = {r["session"]: r for r in episodes(pickle.load(open(cf2p, "rb"))["store"])} if cf2p.exists() else {}
    fail_att = [s for s in ben if s["reward"] != 1]
    a = sum(cf[s["session"]]["reward"] == 1 for s in fail_att)
    fail_cf = [k for k in (s["session"] for s in ben) if cf[k]["reward"] != 1]
    b = sum(cf2[k]["reward"] == 1 for k in fail_cf) if cf2 else 0
    rows.append(("/".join(d.parts[-2:]), len(ben), len(fail_att), a, len(fail_cf), b))
    tot["benign"] += len(ben); tot["after_fail"] += len(fail_att); tot["fail_att_ok_cf"] += a
    tot["fail_cf"] += len(fail_cf); tot["fail_cf_ok_cf2"] += b
for r in rows:
    print(*r)
print(tot, len(rows))
keys = ("run", "benign", "failed_attack", "failed_attack_ok_cf", "failed_cf", "failed_cf_ok_cf2")
json.dump(dict(total=tot, runs=[dict(zip(keys, r)) for r in rows]), open("paper/data/retries.json", "w"), indent=1)
