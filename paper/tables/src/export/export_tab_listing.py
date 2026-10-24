import glob, json, os
R = os.path.expanduser("~/Memloop/results/minja_recovery")
out = {}
for model in ("qwen3-8b", "qwen3-14b", "qwen3-32b"):
    B = f"{R}/{model}-listing"
    dist = json.load(open(f"{B}/distributional.json"))["runs"]
    out[model] = []
    for d in sorted(glob.glob(f"{B}/*/seed*")):
        run = "/".join(d.split("/")[-2:])
        j = lambda f: json.load(open(f"{d}/{f}"))
        ev, st, c = j("eval_attack_B1_B2_recovered_contain_counterfactual.json"), j("recovered_contain_stats.json"), j("compare.json")
        out[model].append(dict(
            run=run, asr={k: v["asr"] for k, v in ev.items()}, memory={k: v["memory"] for k, v in ev.items()},
            target_purchases=c["attack"]["target_purchases"],
            replayed=sum(st["steps_replayed"].values()), steps_after_tau=st["steps_after_tau"],
            seconds=st["seconds"],
            stored={k: dist[run][k]["stored"] for k in ("recovered_contain", "counterfactual", "counterfactual2")}))
print(json.dumps(out, indent=1))
