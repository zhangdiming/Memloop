import glob
import json
import random
from pathlib import Path

B = Path("results/minja_recovery/qwen3-14b-main")
STATES = ["recovered_contain", "loop", "DF", "DP", "B2", "B1", "counterfactual", "empty"]
named = {r["run"]: r["attacker_named"] for r in json.load(open(B / "detector_summary.json"))["runs"]}

runs = {}
for d in sorted(glob.glob(str(B / "*/seed*"))):
    run = "/".join(Path(d).parts[-2:])
    u = {}
    for f in glob.glob(f"{d}/utility_*.json"):
        u[Path(f).stem[len("utility_"):]] = json.load(open(f))
    if "recovered_contain" in u and "DF" in u:
        u["loop"] = u["recovered_contain"] if named[run] else u["DF"]
    runs[run] = u


def per_run(state, task_set, key):
    out = {}
    for run, u in runs.items():
        if state in u:
            e = [x for x in u[state]["episodes"] if x["set"] == task_set]
            out[run] = sum((x["score"] == 1) if key == "success" else x[key] for x in e) / len(e)
    return out


def cluster_ci(diffs, reps=10000, seed=0):
    by_pair = {}
    for run, v in diffs.items():
        by_pair.setdefault(run.split("/")[0], []).append(v)
    pairs, rng, means = list(by_pair.values()), random.Random(seed), []
    for _ in range(reps):
        s = [v for p in (rng.choice(pairs) for _ in pairs) for v in p]
        means.append(sum(s) / len(s))
    means.sort()
    return sum(diffs.values()) / len(diffs), means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


summary = {}
for st in STATES:
    row = {}
    for ts in ("victim", "other"):
        for key in ("success", "score") + (("hit",) if ts == "victim" else ()):
            v = per_run(st, ts, key)
            if not v:
                continue
            row[f"{ts}_{key}"] = sum(v.values()) / len(v)
            if st != "recovered_contain":
                base = per_run("recovered_contain", ts, key)
                common = {r: v[r] - base[r] for r in v if r in base}
                if common:
                    m, lo, hi = cluster_ci(common)
                    row[f"{ts}_{key}_minus_memloop"] = dict(mean=m, lo=lo, hi=hi, runs=len(common))
    mem = [u[st]["memory"] for u in runs.values() if st in u]
    if mem:
        row["memory"] = sum(mem) / len(mem)
        row["runs"] = len(mem)
    summary[st] = row
json.dump(summary, open(B / "utility_summary.json", "w"), indent=1)
for st, row in summary.items():
    if row:
        print(f"{st:18s} runs={row['runs']:2d} mem={row['memory']:5.1f} "
              f"victim succ={row.get('victim_success', float('nan')):.3f} score={row.get('victim_score', float('nan')):.3f} "
              f"asr={row.get('victim_hit', float('nan')):.3f} | other succ={row.get('other_success', float('nan')):.3f} "
              f"score={row.get('other_score', float('nan')):.3f}")
