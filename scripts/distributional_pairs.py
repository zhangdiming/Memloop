import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = json.load(open(ROOT / "results/minja_recovery/qwen3-14b-main/distributional.json"))["runs"]


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
                hi=means[int(0.975 * reps) - 1], runs=len(diffs),
                nonzero_runs=sum(v != 0 for v in diffs.values()))


out = {}
for k in ("benign_success", "stored", "target_buys", "eval_asr"):
    out[k] = dict(rec_minus_cf=cluster_ci({r: v["recovered_contain"][k] - v["counterfactual"][k]
                                           for r, v in RUNS.items()}))
    if all(k in v["counterfactual2"] for v in RUNS.values()):
        out[k]["cf2_minus_cf"] = cluster_ci({r: v["counterfactual2"][k] - v["counterfactual"][k]
                                             for r, v in RUNS.items()})
json.dump(out, open(ROOT / "paper/data/distributional_pairs.json", "w"), indent=1)
for k, v in out.items():
    print(k, {a: f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] nonzero {c['nonzero_runs']}"
              for a, c in v.items()})
