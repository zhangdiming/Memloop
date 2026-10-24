import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MAIN = RES / "minja_recovery/qwen3-14b-main"
SIM = ["r075_f05_d0", "r050_f05_d0", "r025_f05_d0"]
EVAL = "eval_DB_DBs_r075_f05_d0_DBs_r050_f05_d0_DBs_r025_f05_d0.json"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def cluster_ci(diffs, reps=10000, seed=0):
    by_pair = defaultdict(list)
    for run, v in diffs.items():
        by_pair[run.split("/")[0]].append(v)
    pairs, rng, means = list(by_pair.values()), random.Random(seed), []
    for _ in range(reps):
        s = [v for p in (rng.choice(pairs) for _ in pairs) for v in p]
        means.append(sum(s) / len(s))
    means.sort()
    return dict(mean=sum(diffs.values()) / len(diffs), lo=means[int(0.025 * reps)],
                hi=means[int(0.975 * reps) - 1], runs=len(diffs))


def per_run(eps, task_set, key, keep=None):
    e = [x for x in eps if x["set"] == task_set and (keep is None or x["session"] in keep)]
    if not e:
        return None
    return sum((x["score"] == 1) if key == "success" else x[key] for x in e) / len(e)


def main():
    offline = json.load(open(RES / "blamed_delete.json"))
    sim = json.load(open(ROOT / "paper/data/sim_delete.json"))
    util = json.load(open(ROOT / "paper/data/utility.json"))["runs"]
    out = dict(offline={}, eval={}, runs={}, utility={})
    by = defaultdict(list)
    for r in offline:
        key = "llm" if r["detector"] == "llm" else r["detector"][:4]
        by[key].append(r)
    for key, rows in by.items():
        out["offline"][key] = dict(
            n=len(rows), kept=mean([r["kept"] for r in rows]), flagged=mean([r["flagged"] for r in rows]),
            attacker_left=mean([r["attacker_left"] for r in rows]),
            poisoned_left=mean([r["poisoned_left"] for r in rows]),
            attacker_named=sum(r["attacker_named"] for r in rows),
            innocent_named=sum(any(o != "user:attacker" for o in r["origins"]) for r in rows))
    missing = []
    for d in sorted(MAIN.glob("*/seed*")):
        run = "/".join(d.parts[-2:])
        f = d / EVAL
        if not f.exists():
            missing.append(run)
            continue
        ev = json.load(open(f))
        out["runs"][run] = {k: ev[k] for k in ev}
    state = {"llm": "DB", **{t: f"DBs_{t}" for t in SIM}}
    df_sim = defaultdict(dict)
    for r in sim["runs"]:
        df_sim[str(r["recall"])][r["run"]] = r["df"]
    for det, st in state.items():
        rows = [v[st] for v in out["runs"].values() if st in v]
        if not rows:
            continue
        o = [r for r in offline if r["state"] == st]
        g = dict(n=len(rows), asr=mean([r["asr"] for r in rows]), residual_runs=sum(r["asr"] > 0 for r in rows),
                 max=max(r["asr"] for r in rows), memory=mean([r["memory"] for r in rows]),
                 attacker_left=mean([r["attacker_live"] for r in rows]),
                 poisoned_left=mean([r["poisoned_live"] for r in rows]),
                 attacker_named=sum(r["attacker_named"] for r in o))
        if det != "llm":
            rec = str({"r075": 0.75, "r050": 0.5, "r025": 0.25}[det[:4]])
            dfs = df_sim[rec]
            common = {run: out["runs"][run][st]["asr"] - dfs[run]["asr"] for run in out["runs"] if run in dfs}
            g["minus_delete_flagged"] = cluster_ci(common)
        out["eval"][det] = g
    have = []
    for d in sorted(MAIN.glob("*/seed*")):
        run = "/".join(d.parts[-2:])
        f, g = d / "utility_DB.json", d / "utility_counterfactual_db.json"
        if f.exists() and g.exists():
            util.setdefault(run, {})["DB"] = json.load(open(f))
            util[run]["CFdb"] = json.load(open(g))
            have.append(run)
    if have:
        u = dict(runs=len(have), memory=mean([util[r]["DB"]["memory"] for r in have]))
        for st in ("DB", "CFdb"):
            for ts in ("victim", "other"):
                for key in ("success", "score", "hit"):
                    u[f"{st}_{ts}_{key}"] = mean([per_run(util[r][st]["episodes"], ts, key) for r in have])
        for a, b in (("DB", "CFdb"), ("CFdb", "counterfactual")):
            for ts in ("victim", "other"):
                for key in ("success", "score"):
                    u[f"{a}_minus_{b}_{ts}_{key}"] = cluster_ci(
                        {r: per_run(util[r][a]["episodes"], ts, key) - per_run(util[r][b]["episodes"], ts, key)
                         for r in have})
        same = sum({x["session"]: x["instruction"] for x in util[r]["counterfactual"]["episodes"]}.get(x["session"])
                   == x["instruction"] for r in have for x in util[r]["CFdb"]["episodes"])
        u["tasks"], u["tasks_same_instruction_as_c065"] = sum(len(util[r]["CFdb"]["episodes"]) for r in have), same
        out["utility"] = u
        out["utility_runs"] = {r: dict(DB=util[r]["DB"], counterfactual_db=util[r]["CFdb"]) for r in have}
    if missing:
        print("missing evaluations:", missing, file=sys.stderr)
    json.dump(out, open(ROOT / "paper/data/blamed_delete.json", "w"), indent=1)
    print(f"{'detector':>8} | {'kept':>5} {'att.left':>8} {'pois.left':>9} {'named':>5} || "
          f"{'n':>2} {'DB ASR':>7} {'res.runs':>8} {'max':>5} {'DB mem':>6} {'DB-DF ASR [CI]':>24}")
    for det in ("llm", "r075", "r050", "r025"):
        o = out["offline"].get(det, {})
        g = out["eval"].get(det if det == "llm" else det + "_f05_d0", {})
        ci = g.get("minus_delete_flagged")
        print(f"{det:>8} | {o.get('kept', 0):5.1f} {o.get('attacker_left', 0):8.2f} {o.get('poisoned_left', 0):9.2f} "
              f"{o.get('attacker_named', 0):>5} || {g.get('n', 0):2d} {g.get('asr', 0):7.3f} {g.get('residual_runs', 0):8d} "
              f"{g.get('max', 0):5.2f} {g.get('memory', 0):6.1f} "
              + (f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]" if ci else ""))
    if out["utility"]:
        u = out["utility"]
        print(f"utility, {u['runs']} runs, DB memory {u['memory']:.1f}; CF_db instructions equal to the C065 instance's: "
              f"{u['tasks_same_instruction_as_c065']}/{u['tasks']}")
        for st in ("DB", "CFdb"):
            print(f"  {st:5s} victim succ {u[st + '_victim_success']:.3f} score {u[st + '_victim_score']:.3f} "
                  f"hit {u[st + '_victim_hit']:.3f} | other succ {u[st + '_other_success']:.3f} score {u[st + '_other_score']:.3f}")
        for a, b in (("DB", "CFdb"), ("CFdb", "counterfactual")):
            for ts in ("victim", "other"):
                for key in ("score", "success"):
                    c = u[f"{a}_minus_{b}_{ts}_{key}"]
                    print(f"  {a} - {b} {ts:6s} {key:7s} {c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")


if __name__ == "__main__":
    main()
