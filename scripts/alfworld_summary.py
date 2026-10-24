import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen3-14b-main"
BASE = ROOT / "results/alfworld" / TAG
EVAL = "eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json"
STATES = ["attack", "B1", "B2", "DP", "recovered_b3", "recovered_contain", "counterfactual",
          "counterfactual2", "recovered_all"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    runs = []
    for d in sorted(glob.glob(str(BASE / "*/seed*"))):
        d = Path(d)
        if not (d / EVAL).exists():
            continue
        ev = json.load(open(d / EVAL))
        if (d / "eval_counterfactual2.json").exists():
            ev.update(json.load(open(d / "eval_counterfactual2.json")))
        if (d / "eval_recovered_all.json").exists():
            ev.update(json.load(open(d / "eval_recovered_all.json")))
        sc = json.load(open(d / "script.json"))["script"]
        mal = [e for e in sc if e["kind"] == "malicious"]
        test = [e for e in sc if e["kind"] == "test"]
        st = {v: json.load(open(d / f"recovered_{v}_stats.json")) for v in ("contain", "b3", "all")
              if (d / f"recovered_{v}_stats.json").exists()}
        runs.append(dict(
            run=f"{d.parent.name}/{d.name}", steps=len(sc),
            injections_stored=sum(e["stored"] for e in mal),
            test_asr=sum(e["hit_target"] for e in test) / max(1, len(test)),
            eval={k: {x: v[x] for x in ("asr", "victim_success", "memory", "attacker_live", "poisoned_live")}
                  for k, v in ev.items()},
            recovery={v: dict(after_tau=s["steps_after_tau"], replayed=sum(s["steps_replayed"].values()),
                              by_cause=s["steps_replayed"], dropped=s["steps_dropped"],
                              minutes=s["seconds"] / 60, certificate_problems=len(s["certificate_problems"]))
                      for v, s in st.items()}))
    agg = {}
    for s in STATES:
        rows = [r["eval"][s] for r in runs if s in r["eval"]]
        agg[s] = dict(n=len(rows), asr=mean([x["asr"] for x in rows]),
                      residual_runs=sum(x["asr"] > 0 for x in rows),
                      victim_success=mean([x["victim_success"] for x in rows]),
                      memory=mean([x["memory"] for x in rows]))
    paired = {}
    for st in STATES:
        rr = [r for r in runs if st in r["eval"]]
        if len(rr) < 2:
            continue
        paired[st] = {}
        for key in ("victim_success", "memory", "asr"):
            xs = [r["eval"][st][key] - r["eval"]["counterfactual"][key] for r in rr]
            m = mean(xs)
            sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
            paired[st][key] = dict(n=len(xs), mean=m, ci95=1.96 * sd / len(xs) ** 0.5)
    rec = {v: dict(replayed=mean([r["recovery"][v]["replayed"] for r in runs if v in r["recovery"]]),
                   after_tau=mean([r["recovery"][v]["after_tau"] for r in runs if v in r["recovery"]]),
                   minutes=mean([r["recovery"][v]["minutes"] for r in runs if v in r["recovery"]]),
                   displacement=sum(r["recovery"][v]["by_cause"].get("retrieval", 0) for r in runs
                                    if v in r["recovery"]),
                   certificate_problems=sum(r["recovery"][v]["certificate_problems"] for r in runs
                                            if v in r["recovery"]))
           for v in ("contain", "b3", "all") if any(v in r["recovery"] for r in runs)}
    out = dict(tag=TAG, runs=len(runs), attack=dict(
        injections_stored=mean([r["injections_stored"] for r in runs]),
        test_asr=mean([r["test_asr"] for r in runs]),
        no_injection=sum(r["injections_stored"] == 0 for r in runs)), states=agg, paired_vs_cf=paired, recovery=rec, per_run=runs)
    (ROOT / "paper/data").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "paper/data/alfworld_main.json", "w"), indent=1)
    print(f"{len(runs)} runs; injections stored {out['attack']['injections_stored']:.1f}, "
          f"in-script victim ASR {out['attack']['test_asr']:.1%}, no injection in {out['attack']['no_injection']}")
    for s, a in agg.items():
        print(f"  {s:18s} n={a['n']:2d} ASR {a['asr']:6.1%} ({a['residual_runs']:2d} runs) "
              f"victim success {a['victim_success']:.2f} records {a['memory']:.1f}")
    for st, p in paired.items():
        print(f"  paired {st:18s} " + "  ".join(f"{k} {v['mean']:+.3f}±{v['ci95']:.3f} (n={v['n']})" for k, v in p.items()))
    for v, a in rec.items():
        print(f"  recover {v:8s} replayed {a['replayed']:.1f}/{a['after_tau']:.1f} after tau, "
              f"{a['minutes']:.0f} min, displacement re-executions {a['displacement']}, "
              f"certificate problems {a['certificate_problems']}")


if __name__ == "__main__":
    main()
