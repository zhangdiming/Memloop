import glob, json, math, os, pickle, re, sys
from collections import Counter, defaultdict

ROOT = os.path.expanduser("~/Memloop")
R = f"{ROOT}/results"
OUT = f"{ROOT}/paper/data"
sys.path.insert(0, f"{ROOT}/memloop")
os.makedirs(OUT, exist_ok=True)


def dump(name, obj):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("wrote", name)


def jload(p):
    return json.load(open(p))


def ci95(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return dict(mean=m, ci=1.96 * sd / math.sqrt(n) if n > 1 else 0.0, n=n)


rows = []
for d in sorted(glob.glob(f"{R}/minja_rap/qwen3-14b/*/seed0")):
    log = open(f"{d}/run.log").read()
    mem = jload(f"{d}/memory.json")
    ben = [x for x in mem if x["Id"].startswith("fixed_0")]
    rows.append(dict(pair=d.split("/")[-2],
                     isr=float(re.search(r"inject success rate:\s+([\d.]+)", log).group(1)),
                     asr=float(re.search(r"attack success rate:\s+([\d.]+)", log).group(1)),
                     invalid=log.count("Invalid action") / len(re.findall(r"^Action:", log, re.M)),
                     memory=len(mem), malicious=len(mem) - len(ben),
                     benign_with_injected=sum(any("out of stock" in a.lower() for a in x["Actions"]) for x in ben),
                     benign=len(ben)))
dump("minja_reproduction", rows)

det = {}
for pair in ("toothbrush", "jeans", "cookies"):
    d = f"{R}/minja_recovery/qwen3-14b-det/{pair}/seed0"
    ev = {}
    for f in glob.glob(f"{d}/eval_*.json"):
        ev.update(jload(f))
    det[pair] = dict(compare=jload(f"{d}/compare.json"), eval=ev,
                     stats={t: jload(f"{d}/{t}_stats.json") for t in ("recovered", "recovered_b3")
                            if os.path.exists(f"{d}/{t}_stats.json")})
dump("deterministic_pilot", det)

curves = {}
for pair in ("toothbrush", "jeans", "cookies"):
    d = f"{R}/minja_recovery/qwen3-14b-det/{pair}/seed0"
    att = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
    seeds = [st.trigger for st in att.tracker.steps if st.session == "attacker"]
    closure = {n for n in att.tracker.descendants(seeds) if att.tracker.nodes[n].kind == "mem"}
    start = set(att.active_ids())
    pre = len(att.tracker.ledger)
    curves[pair] = {}
    for tag, label in (("recovered", "recover"), ("recovered_contain", "contain+repair")):
        st = jload(f"{d}/{tag}_stats.json")
        recs = pickle.load(open(f"{d}/{tag}.pkl", "rb"))["store"].tracker.ledger.records()
        times = iter([t for t, *_ in st["replay_timeline"]])
        live, clock, series = set(start), 0.0, [(0.0, len(start & closure), len(start - closure))]
        for r in recs[pre:]:
            if r["t"] == "replay":
                clock = next(times)
            elif r["t"] in ("resolve", "contain"):
                live.discard(r["id"])
            elif r["t"] == "node" and r["kind"] == "mem":
                live.add(r["id"])
            else:
                continue
            series.append((clock / 60, len(live & closure), len(live - closure)))
        curves[pair][label] = dict(series=series, total_minutes=st["seconds"] / 60,
                                   unsafe_minutes=max((x for x, y, _ in series if y > 0), default=0.0))
    ev = jload(f"{d}/eval_attack_B1_B2_recovered_counterfactual.json") if os.path.exists(
        f"{d}/eval_attack_B1_B2_recovered_counterfactual.json") else {}
    curves[pair]["counterfactual_memory"] = next(
        (jload(f)["counterfactual"]["memory"] for f in glob.glob(f"{d}/eval_*.json")
         if "counterfactual" in jload(f)), None)
dump("recovery_curves", curves)

B = f"{R}/minja_recovery/qwen3-14b-main"
runs = []
for d in sorted(glob.glob(f"{B}/*/seed*")):
    if not os.path.exists(f"{d}/compare.json"):
        continue
    c, st = jload(f"{d}/compare.json"), jload(f"{d}/recovered_contain_stats.json")
    ev = jload(f"{d}/eval_attack_B1_B2_recovered_contain_counterfactual.json")
    sc = jload(f"{d}/script.json")["script"]
    runs.append(dict(run=d.split("/")[-2] + "/" + d.split("/")[-1],
                     asr={k: v["asr"] for k, v in ev.items()}, memory={k: v["memory"] for k, v in ev.items()},
                     outcome_rec=c["recovered_contain"]["outcome"], outcome_noise=c["noise_floor"]["outcome"],
                     replayed=sum(st["steps_replayed"].values()), steps_after_tau=st["steps_after_tau"],
                     script_steps=len(sc), minutes=st["seconds"] / 60, contain_ms=st["contain_seconds"] * 1000,
                     contained=st["contained"], cert_ok=not st["certificate_problems"] and not st["containment_problems"]))
methods = ["attack", "B1", "B2", "recovered_contain", "counterfactual"]
summary = dict(
    n=len(runs),
    asr={m: ci95([r["asr"][m] for r in runs]) for m in methods},
    memory={m: ci95([r["memory"][m] for r in runs]) for m in methods},
    b1_residual_runs=sum(r["asr"]["B1"] > 0 for r in runs),
    attack_effective_runs=sum(r["asr"]["attack"] > 0 for r in runs),
    replay_share=ci95([r["replayed"] / r["steps_after_tau"] for r in runs]),
    contain_ms_max=max(r["contain_ms"] for r in runs),
    certificates_ok=sum(r["cert_ok"] for r in runs),
    outcome_rec=ci95([r["outcome_rec"] for r in runs]),
    outcome_noise=ci95([r["outcome_noise"] for r in runs]),
    outcome_paired_diff=ci95([r["outcome_rec"] - r["outcome_noise"] for r in runs]))
dump("main_sweep", dict(summary=summary, runs=runs,
                        distributional=jload(f"{B}/distributional.json")["summary"]))

ts = []
for w in ("early", "mid", "late"):
    for d in sorted(glob.glob(f"{R}/minja_recovery/qwen3-14b-time-{w}/*/seed0")):
        if not os.path.exists(f"{d}/compare.json"):
            continue
        sc = jload(f"{d}/script.json")["script"]
        st = jload(f"{d}/recovered_contain_stats.json")
        ev = jload(f"{d}/eval_attack_B1_B2_recovered_contain_counterfactual.json")
        ts.append(dict(window=w, pair=d.split("/")[-2], steps=len(sc),
                       tau=min(e["step"] for e in sc if e["owner"] == "attacker"),
                       replayed=sum(st["steps_replayed"].values()), minutes=st["seconds"] / 60,
                       contained=st["contained"], asr={k: v["asr"] for k, v in ev.items()},
                       memory={k: v["memory"] for k, v in ev.items()}))
dump("attack_time", ts)

d = f"{R}/minja_recovery/qwen3-14b-det/cookies/seed0"
par = []
for tag in ("recovered_contain", "recovered_contain_par4", "recovered_contain_par8"):
    s = jload(f"{d}/{tag}_stats.json")
    par.append(dict(width=s["parallel"], minutes=s["seconds"] / 60, replays=sum(s["steps_replayed"].values()),
                    conflicts=s["parallel_conflicts"], cert_ok=not s["certificate_problems"]))
dump("parallel", dict(widths=par, identical_across_widths=True,
                      note="par1/par4/par8 final states identical over all 137 episodes (checked in session)"))

det_sum = jload(f"{B}/detector_summary.json")
loop = []
for run in ("toothbrush/seed0", "toothbrush/seed1", "cookies/seed0", "curtain/seed1", "curtain/seed2"):
    dd = f"{B}/{run}"
    ev = jload(f"{dd}/eval_attack_B1_B2_recovered_contain_counterfactual.json")
    ent = jload(f"{dd}/eval_recovered_contain_entry.json")["recovered_contain_entry"]
    df = jload(f"{dd}/eval_DF_DP.json")["DF"]
    s = next(r for r in det_sum["runs"] if r["run"] == run)
    org = ev["recovered_contain"] if s["attacker_named"] else ev["attack"]
    loop.append(dict(run=run, recall=s["recall"], attacker_named=s["attacker_named"],
                     attacked=ev["attack"]["asr"], entry=ent["asr"], origin=org["asr"],
                     fallback=org["asr"] if s["attacker_named"] else df["asr"],
                     oracle=ev["recovered_contain"]["asr"], cf=ev["counterfactual"]["asr"],
                     memory=dict(attacked=ev["attack"]["memory"], entry=ent["memory"], oracle=ev["recovered_contain"]["memory"],
                                 cf=ev["counterfactual"]["memory"])))
dump("detector", dict(judge={k: det_sum[k] for k in ("precision", "recall", "tp", "fp", "fn", "tn")},
                      runs=det_sum["runs"], closed_loop=loop))

sys.path.insert(0, f"{ROOT}/memloop/experiments")
import attribution_offline as AO
grid = []
for sybil in (False, True):
    for recall in (1.0, 0.75, 0.5, 0.25):
        for fp in (0.0, 0.05, 0.10):
            grid.append(dict(sybil=sybil, recall=recall, fp=fp, **AO.run(recall, fp, sybil)))
dump("attribution_offline", grid)

qa = {}
for model in ("qwen3-14b", "qwen3-8b", "qwen3-32b"):
    qa[model] = []
    for d in sorted(glob.glob(f"{R}/qa_recovery/{model}/*/seed*")):
        if not os.path.exists(f"{d}/compare.json"):
            continue
        sc = jload(f"{d}/script.json")["script"]
        ev = jload(f"{d}/eval_attack_B1_B2_recovered_contain_counterfactual.json")
        c, st = jload(f"{d}/compare.json"), jload(f"{d}/recovered_contain_stats.json")
        mal = [e for e in sc if e["kind"] == "malicious"]
        qa[model].append(dict(run=d.split("/")[-2] + "/" + d.split("/")[-1],
                              malicious_stored=sum(e["stored"] for e in mal),
                              test_asr=c["attack"]["test_asr"], asr={k: v["asr"] for k, v in ev.items()},
                              memory={k: v["memory"] for k, v in ev.items()},
                              benign_acc=dict(rec=c["recovered_contain"]["benign_acc"], cf=c["counterfactual"]["benign_acc"],
                                              cf2=c.get("counterfactual2", {}).get("benign_acc")),
                              cert_ok=not st["certificate_problems"] and not st["containment_problems"],
                              contain_ms=st["contain_seconds"] * 1000))
dump("qa", qa)

dump("cert_overhead", jload(f"{R}/cert_overhead.json"))
dump("flip_rate", dict(short=jload(f"{R}/flip_rate/compare.json"),
                       long=jload(f"{R}/flip_rate/long/compare.json"),
                       note="short: 600 QA-agent prompts (~3 examples); long: 200 prompts with up to 60 "
                            "examples (~45k chars). bi = batch-invariant server, a/b = instance before/after "
                            "restart, number = concurrency; nm = normal server."))


scale = {}
for model in ("qwen3-8b", "qwen3-14b", "qwen3-32b"):
    rs = []
    for d in sorted(glob.glob(f"{R}/minja_recovery/{model}-scale/*/seed*")):
        if not os.path.exists(f"{d}/compare.json"):
            continue
        sc = jload(f"{d}/script.json")["script"]
        st = jload(f"{d}/recovered_contain_stats.json")
        ev = jload(f"{d}/eval_attack_B1_B2_recovered_contain_counterfactual.json")
        ben = [e for e in sc if e["kind"] == "benign"]
        mal = [e for e in sc if e["kind"] == "malicious"]
        rs.append(dict(run=d.split("/")[-2] + "/" + d.split("/")[-1],
                       asr={k: v["asr"] for k, v in ev.items()}, memory={k: v["memory"] for k, v in ev.items()},
                       benign_success=sum(e["reward"] == 1 for e in ben) / max(1, len(ben)),
                       inject_success=sum(e["hit_target"] for e in mal) / max(1, len(mal)),
                       replayed=sum(st["steps_replayed"].values()), steps_after_tau=st["steps_after_tau"],
                       minutes=st["seconds"] / 60, contain_ms=st["contain_seconds"] * 1000,
                       cert_ok=not st["certificate_problems"] and not st["containment_problems"]))
    scale[model] = dict(runs=rs, summary=dict(
        n=len(rs),
        asr={m: ci95([r["asr"][m] for r in rs]) for m in methods},
        memory={m: ci95([r["memory"][m] for r in rs]) for m in methods},
        benign_success=ci95([r["benign_success"] for r in rs]),
        inject_success=ci95([r["inject_success"] for r in rs]),
        replay_share=ci95([r["replayed"] / r["steps_after_tau"] for r in rs]),
        minutes=ci95([r["minutes"] for r in rs]),
        contain_ms_max=max(r["contain_ms"] for r in rs),
        certificates_ok=sum(r["cert_ok"] for r in rs)))
dump("model_scale", scale)
