import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")


def load(p):
    return json.load(open(p))


rows = []
for f in sorted(glob.glob(str(root / "amem_evo/*/*/*/*/compare.json"))):
    d = load(f)
    m, b = d["recovered_contain"], d["recovered_contain_b3"]
    rows.append(dict(run=str(Path(f).parent.relative_to(root)),
                     exact=m["exact"] == 1.0 and m["memory_equal"],
                     b3_diff=b["differ_after_tau"], after=b["after_tau"], b3_jac=b["memory_jaccard"]))
print(f"memory evolution: {len(rows)} runs")
print(f"  MemLoop exact in every episode and final memory: {sum(r['exact'] for r in rows)}/{len(rows)}")
div = [r for r in rows if r["b3_diff"] > 0]
worst = max(rows, key=lambda r: r["b3_diff"])
print(f"  Edge-Replay diverged in {len(div)} runs, {sum(r['b3_diff'] for r in rows)}"
      f"/{sum(r['after'] for r in rows)} later benign episodes;"
      f" worst {worst['b3_diff']}/{worst['after']} ({worst['run']});"
      f" min Jaccard {min(r['b3_jac'] for r in rows):.2f}")
for g in ("w55", "w80"):
    rr = [r for r in rows if f"/{g}/" in "/" + r["run"]]
    print(f"  grid {g}: {len(rr)} runs, {sum(r['after'] for r in rr)} later benign episodes")

print("served during repair:")
for f in sorted(glob.glob(str(root / "minja_recovery/qwen3-14b-*/*/*/compare_counterfactual_during*.json"))):
    d = load(f)
    (k, v), = [(k, v) for k, v in d.items() if k != "counterfactual"]
    print(f"  {Path(f).parent.relative_to(root)}  {k}: exact={v['exact']:.3f}"
          f" jaccard={v['memory_jaccard']:.3f} victim_asr={v['test_asr']:.2f}")
