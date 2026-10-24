import glob, json, math, os
rows = []
for d in sorted(glob.glob("results/minja_recovery/qwen3-14b-main/*/seed*")):
    p = f"{d}/compare.json"
    if not os.path.exists(p):
        continue
    c = json.load(open(p))
    rows.append((d.split("/")[-2] + "/" + d.split("/")[-1], c["recovered_contain"]["outcome"],
                 c["noise_floor"]["outcome"], c["attack"]["outcome"], c["recovered_contain"]["episodes"]))
diffs = [r - nf for _, r, nf, _, _ in rows]
n = len(diffs); m = sum(diffs) / n
sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
print(f"{'run':22s} rec-vs-cf  cf-vs-cf2   diff  attacked-vs-cf  episodes")
for (name, r, nf, a, e), d in zip(rows, diffs):
    print(f"{name:22s} {r:9.3f} {nf:10.3f} {d:+6.3f} {a:15.3f} {e:9d}")
print(f"paired diff mean {m:+.3f}, 95% CI +/- {1.96 * sd / math.sqrt(n):.3f} (n={n}); "
      f"runs with rec < noise floor: {sum(x < 0 for x in diffs)}/{n}")
