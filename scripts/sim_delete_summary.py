import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MAIN = RES / "minja_recovery/qwen3-14b-main"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    offline = json.load(open(RES / "sim_delete.json"))
    out = dict(offline={}, draw0={}, runs=[])
    by = defaultdict(list)
    for r in offline:
        by[r["recall"]].append(r)
    for rec, rows in sorted(by.items(), reverse=True):
        out["offline"][str(rec)] = dict(
            n=len(rows), df_kept=mean([r["kept"] for r in rows]),
            df_attacker_left=mean([r["attacker_left"] for r in rows]),
            df_poisoned_left=mean([r["poisoned_left"] for r in rows]),
            named=sum(r["oracle_seeds"] for r in rows))
    missing = []
    for rec in sorted(by, reverse=True):
        rows = [r for r in by[rec] if r["draw"] == 0]
        df_asr, ml_asr, df_mem, ml_mem, df_res, ml_res = [], [], [], [], 0, 0
        for r in rows:
            d = MAIN / r["run"]
            f = d / "eval_DFs_r075_f05_d0_DFs_r050_f05_d0_DFs_r025_f05_d0.json"
            if not f.exists():
                missing.append(r["run"])
                continue
            df = json.load(open(f))[f"DFs_{r['tag']}"]
            if r["oracle_seeds"]:
                ml = json.load(open(d / "eval_attack_B1_B2_recovered_contain_counterfactual.json"))["recovered_contain"]
                how = "oracle"
            else:
                ml = df
                how = "fallback"
            entry = None
            if how == "fallback":
                entry = json.load(open(d / f"eval_recovered_contain_sim_{r['tag']}.json"))[f"recovered_contain_sim_{r['tag']}"]
            df_asr.append(df["asr"]); ml_asr.append(ml["asr"])
            df_mem.append(df["memory"]); ml_mem.append(ml["memory"])
            df_res += df["asr"] > 0; ml_res += ml["asr"] > 0
            out["runs"].append(dict(run=r["run"], recall=rec, df=df, memloop=ml, memloop_seeds=how, entry=entry))
        out["draw0"][str(rec)] = dict(
            n=len(df_asr), df_asr=mean(df_asr), df_residual_runs=df_res, df_max=max(df_asr, default=0),
            df_memory=mean(df_mem), memloop_asr=mean(ml_asr), memloop_residual_runs=ml_res,
            memloop_memory=mean(ml_mem))
    out["camera2_round2"] = {}
    for t in ("r075_f05_d0", "r050_f05_d0"):
        d = MAIN / "camera/seed2"
        f = d / f"eval_recovered_contain_sim_{t}_r2.json"
        if f.exists():
            st = json.load(open(d / f"recovered_contain_sim_{t}_r2_stats.json"))
            out["camera2_round2"][t] = dict(json.load(open(f))[f"recovered_contain_sim_{t}_r2"],
                                            steps_dropped=st["steps_dropped"], minutes=st["seconds"] / 60)
    if missing:
        print("missing evaluations:", sorted(set(missing)), file=sys.stderr)
    json.dump(out, open(ROOT / "paper/data/sim_delete.json", "w"), indent=1)
    print(f"{'recall':>6} | {'DF kept':>7} {'att.left':>8} {'pois.left':>9} {'named':>6} || "
          f"{'n':>2} {'DF ASR':>7} {'res.runs':>8} {'max':>5} {'DF mem':>6} | {'ML ASR':>6} {'res.runs':>8} {'ML mem':>6}")
    for rec in out["offline"]:
        o, g = out["offline"][rec], out["draw0"].get(rec, {})
        print(f"{rec:>6} | {o['df_kept']:7.1f} {o['df_attacker_left']:8.1f} {o['df_poisoned_left']:9.1f} "
              f"{o['named']:>3}/{o['n']} || {g.get('n', 0):2d} {g.get('df_asr', 0):7.1%} "
              f"{g.get('df_residual_runs', 0):8d} {g.get('df_max', 0):5.0%} {g.get('df_memory', 0):6.1f} | "
              f"{g.get('memloop_asr', 0):6.1%} {g.get('memloop_residual_runs', 0):8d} {g.get('memloop_memory', 0):6.1f}")


if __name__ == "__main__":
    main()
