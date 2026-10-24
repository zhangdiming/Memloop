import glob, json, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "memloop"), str(ROOT / "memloop/experiments")]
from alfworld_recovery import episodes, ATTACKER
TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen3-14b-main"
STATES = ["attack", "counterfactual", "counterfactual2", "recovered_contain", "recovered_b3", "recovered_all"]


def main():
    out = []
    for d in sorted(glob.glob(str(ROOT / "results/alfworld" / TAG / "*/seed*"))):
        d = Path(d)
        sc = json.load(open(d / "script.json"))["script"]
        tau = next(i for i, e in enumerate(sc) if e["owner"] == ATTACKER)
        pre = {e["session"] for e in sc[:tau]}
        row = dict(run=f"{d.parent.name}/{d.name}")
        for s in STATES:
            if not (d / f"{s}.pkl").exists():
                continue
            ep = episodes(pickle.load(open(d / f"{s}.pkl", "rb"))["store"])
            for seg, sel in (("pre", lambda r: r["session"] in pre), ("post", lambda r: r["session"] not in pre)):
                rr = [r for r in ep if sel(r) and not r["session"].startswith("eval")]
                row[f"{s}/{seg}/n"] = len(rr)
                row[f"{s}/{seg}/success"] = sum(r["reward"] == 1 for r in rr) / max(1, len(rr))
                row[f"{s}/{seg}/stored"] = sum(r["mem"] is not None for r in rr)
        out.append(row)
    json.dump(out, open(ROOT / "paper/data/alfworld_gap_locate.json", "w"), indent=1)
    keys = sorted({k for r in out for k in r if k != "run"})
    for k in keys:
        xs = [r[k] for r in out if k in r]
        print(f"{k:34s} n={len(xs):2d} mean {sum(xs)/len(xs):.3f}")


if __name__ == "__main__":
    main()
