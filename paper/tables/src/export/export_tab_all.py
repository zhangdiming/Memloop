import json, os, sys
M = "results/minja_recovery/qwen3-14b-main"
VICTIMS = [p["victim"].replace(" ", "_") for p in json.load(open("baselines/MINJA/rap/victim_target_pair/victim_target.json"))]
out = []
for v in sorted(VICTIMS):
    d = f"{M}/{v}/seed0"
    j = lambda f: json.load(open(f"{d}/{f}"))
    c, ev = j("compare.json"), j("eval_attack_B1_B2_recovered_contain_counterfactual.json")
    row = dict(run=f"{v}/seed0")
    for tag, stats in (("selective", "recovered_contain"), ("full", "recovered_all")):
        s = j(f"{stats}_stats.json")
        row[tag] = dict(replayed=sum(s["steps_replayed"].values()), after_tau=s["steps_after_tau"],
                        exact=c[stats]["exact"])
    row["selective"]["memory"] = ev["recovered_contain"]["memory"]
    row["full"]["memory"] = j("eval_recovered_all.json")["recovered_all"]["memory"]
    row["cf2_exact"] = c["noise_floor"]["exact"]
    out.append(row)
json.dump(out, sys.stdout, indent=1)
