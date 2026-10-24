import glob, json, math, os, sys
base = sys.argv[1] if len(sys.argv) > 1 else "results/minja_recovery/qwen3-14b-main"
rows = []
for d in sorted(glob.glob(f"{base}/*/seed*")):
    if not os.path.exists(f"{d}/compare.json"):
        continue
    c = json.load(open(f"{d}/compare.json"))
    ev = json.load(open(f"{d}/eval_attack_B1_B2_recovered_contain_counterfactual.json"))
    st = json.load(open(f"{d}/recovered_contain_stats.json"))
    pair, seed = d.split("/")[-2], d.split("/")[-1]
    rows.append(dict(
        run=f"{pair}/{seed}",
        asr={k: ev[k]["asr"] for k in ev}, mem={k: ev[k]["memory"] for k in ev},
        out_rec=c["recovered_contain"]["outcome"], out_att=c["attack"]["outcome"],
        out_nf=c.get("noise_floor", {}).get("outcome"),
        jac_rec=c["recovered_contain"]["memory_jaccard"], jac_nf=c.get("noise_floor", {}).get("memory_jaccard"),
        test_asr_att=c["attack"]["test_asr"], test_asr_rec=c["recovered_contain"]["test_asr"],
        replayed=sum(st["steps_replayed"].values()), after=st["steps_after_tau"],
        minutes=st["seconds"] / 60, contain_ms=st["contain_seconds"] * 1000,
        cert=not st["certificate_problems"] and not st["containment_problems"]))
methods = ["attack", "B1", "B2", "recovered_contain", "counterfactual"]
print(f"{'run':22s} " + " ".join(f"{m[:9]:>9s}" for m in methods) + " | memory: " +
      " ".join(f"{m[:6]:>6s}" for m in methods) + " | outcome rec/nf  replay  min  cert")
for r in rows:
    print(f"{r['run']:22s} " + " ".join(f"{r['asr'][m]:9.0%}" for m in methods) + " |         " +
          " ".join(f"{r['mem'][m]:6d}" for m in methods) +
          f" | {r['out_rec']:.2f}/{(r['out_nf'] or 0):.2f}  {r['replayed']:3d}/{r['after']:3d} {r['minutes']:5.0f}  {'ok' if r['cert'] else 'FAIL'}")
def ci(xs):
    n = len(xs); m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0
    return m, 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")
if rows:
    print(f"\nn={len(rows)} runs; mean ± 95% CI")
    for m in methods:
        a, ea = ci([r["asr"][m] for r in rows]); b, eb = ci([r["mem"][m] for r in rows])
        print(f"  {m:18s} residual ASR {a:5.1%} ± {ea:4.1%}   memory {b:5.1f} ± {eb:4.1f}")
    for k, lab in (("out_rec", "outcome agreement recovered vs cf"), ("out_nf", "outcome agreement cf vs cf2 (noise floor)"),
                   ("jac_rec", "memory Jaccard recovered vs cf"), ("jac_nf", "memory Jaccard cf vs cf2 (noise floor)"),
                   ("out_att", "outcome agreement attacked vs cf")):
        xs = [r[k] for r in rows if r[k] is not None]
        if xs:
            a, e = ci(xs); print(f"  {lab:45s} {a:.3f} ± {e:.3f}")
    a, e = ci([r["replayed"] / r["after"] for r in rows]); print(f"  replayed share of steps after tau          {a:.2f} ± {e:.2f}")
    print(f"  containment time: max {max(r['contain_ms'] for r in rows):.1f} ms; certificates ok: {sum(r['cert'] for r in rows)}/{len(rows)}")
