import glob, json, math, os, pickle, sys
sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))

base = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Memloop/results/minja_recovery/qwen3-14b-main")


def stats(path):
    tr = pickle.load(open(path, "rb"))["store"].tracker
    ben = tst = ben_ok = tst_hit = stored = target_buys = 0
    for st in tr.steps:
        if st.dropped or st.session == "attacker":
            continue
        kind = tr.nodes[st.trigger].meta["kind"]
        prods = {tr.nodes[p].kind: tr.nodes[p] for p in st.products}
        out = json.loads(prods["out"].text)
        stored += "mem" in prods
        target_buys += bool(out["hit_target"]) and "act" in prods
        if kind == "benign":
            ben += 1; ben_ok += out["reward"] == 1
        elif kind == "test":
            tst += 1; tst_hit += bool(out["hit_target"])
    return dict(benign_success=ben_ok / max(1, ben), test_asr=tst_hit / max(1, tst),
                stored=stored, target_buys=target_buys)


def ci(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, 1.96 * sd / math.sqrt(n)


runs = []
for d in sorted(glob.glob(f"{base}/*/seed*")):
    if not os.path.exists(f"{d}/compare.json"):
        continue
    s = {t: stats(f"{d}/{t}.pkl") for t in ("recovered_contain", "counterfactual", "counterfactual2")}
    ev = json.load(open(f"{d}/eval_attack_B1_B2_recovered_contain_counterfactual.json"))
    s["recovered_contain"]["eval_asr"] = ev["recovered_contain"]["asr"]
    s["counterfactual"]["eval_asr"] = ev["counterfactual"]["asr"]
    s["recovered_contain"]["live_memory"] = ev["recovered_contain"]["memory"]
    s["counterfactual"]["live_memory"] = ev["counterfactual"]["memory"]
    runs.append((d.split("/")[-2] + "/" + d.split("/")[-1], s))

print(f"n = {len(runs)} runs (paired by run; mean difference with 95% CI)\n")
print(f"{'statistic':18s} {'recovered':>10s} {'cf':>8s} {'cf2':>8s}   {'rec - cf':>17s}   {'cf2 - cf (noise)':>17s}")
summary = {}
for k in ("benign_success", "test_asr", "stored", "target_buys", "eval_asr", "live_memory"):
    rec = [s["recovered_contain"][k] for _, s in runs]
    cf = [s["counterfactual"][k] for _, s in runs]
    cf2 = [s["counterfactual2"].get(k) for _, s in runs]
    dm, de = ci([a - b for a, b in zip(rec, cf)])
    line = f"{k:18s} {sum(rec)/len(rec):10.3f} {sum(cf)/len(cf):8.3f} "
    if None not in cf2:
        nm, ne = ci([a - b for a, b in zip(cf2, cf)])
        line += f"{sum(cf2)/len(cf2):8.3f}   {dm:+8.3f} ± {de:6.3f}   {nm:+8.3f} ± {ne:6.3f}"
        summary[k] = dict(rec_minus_cf=(dm, de), cf2_minus_cf=(nm, ne))
    else:
        line += f"{'-':>8s}   {dm:+8.3f} ± {de:6.3f}   {'(not evaluated)':>17s}"
        summary[k] = dict(rec_minus_cf=(dm, de))
    print(line)
json.dump(dict(n=len(runs), summary=summary, runs=dict(runs)),
          open(f"{base}/distributional.json", "w"), indent=1)
print(f"\nsaved {base}/distributional.json")
