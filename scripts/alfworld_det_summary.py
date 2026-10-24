import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/alfworld/qwen3-14b-det"
rows = []
for d in sorted(glob.glob(str(BASE / "*/seed0"))):
    d = Path(d)
    if not (d / "compare.json").exists():
        continue
    c = json.load(open(d / "compare.json"))
    st = {v: json.load(open(d / f"recovered_{v}_stats.json")) for v in ("contain", "b3")}
    rows.append(dict(run=d.parent.name, episodes=c["attack"]["episodes"],
                     **{f"{k}_exact": c[k]["exact"] for k in ("attack", "recovered_contain", "recovered_b3")},
                     **{f"{k}_memory_equal": c[k]["memory_equal"] for k in ("attack", "recovered_contain", "recovered_b3")},
                     attack_jaccard=c["attack"]["memory_jaccard"], attack_test_asr=c["attack"]["test_asr"],
                     after_tau=st["contain"]["steps_after_tau"],
                     replayed=sum(st["contain"]["steps_replayed"].values()),
                     displacement=st["contain"]["steps_replayed"].get("retrieval", 0),
                     minutes=st["contain"]["seconds"] / 60,
                     certificate_problems=len(st["contain"]["certificate_problems"])))
json.dump(rows, open(ROOT / "paper/data/alfworld_det.json", "w"), indent=1)
for r in rows:
    print(f"{r['run']:10s} ep {r['episodes']:3d} attack {r['attack_exact']:.2f} | MemLoop {r['recovered_contain_exact']:.2f} "
          f"mem={r['recovered_contain_memory_equal']} | Edge-Replay {r['recovered_b3_exact']:.2f} | "
          f"replayed {r['replayed']}/{r['after_tau']} disp {r['displacement']} {r['minutes']:.0f} min")
print(f"{len(rows)} runs; MemLoop exact in {sum(r['recovered_contain_exact'] == 1 and r['recovered_contain_memory_equal'] for r in rows)}, "
      f"Edge-Replay in {sum(r['recovered_b3_exact'] == 1 and r['recovered_b3_memory_equal'] for r in rows)}; "
      f"attacked agreement {min(r['attack_exact'] for r in rows):.2f}-{max(r['attack_exact'] for r in rows):.2f}")
