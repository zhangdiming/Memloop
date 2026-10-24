import json, os, sys
from pathlib import Path

R = Path("results")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def run(d, keys):
    c = json.load(open(d / "compare.json"))
    st = {t: json.load(open(d / f"{t}_stats.json")) for t in ("recovered_contain", "recovered_contain_b3")}
    a = c["attack"]
    row = dict(zip(keys, d.parts[-len(keys):]))
    row["attack"] = {k: a[k] for k in ("episodes", "exact", "after_tau", "differ_after_tau", "memory",
                                       "memory_cf", "test_asr", "rewrites") if k in a}
    for tag, name in (("recovered_contain", "memloop"), ("recovered_contain_b3", "b3")):
        m = c[tag]
        row[name] = dict(differ_after_tau=m["differ_after_tau"], memory_equal=m["memory_equal"],
                         memory_jaccard=m["memory_jaccard"], replayed=st[tag]["steps_replayed"])
    ev = list(d.glob("eval_*.json"))
    if ev:
        row["eval"] = {s: {k: v[k] for k in ("hits", "n", "memory")} for s, v in json.load(open(ev[0])).items()}
    return row


for name, sub, pattern, keys in (
        ("qa_amem", "amem_qa", "k*/seed*", ("k", "seed")),
        ("qa_amem_late", "amem_late", "k*/ov*/seed*", ("k", "share", "seed")),
        ("qa_amem_evo", "amem_evo", "w*/k*/ov*/seed*", ("window", "k", "share", "seed")),
        ("qa_disp_stress", "disp_stress", "*/k*/ov*/*/seed*", ("key", "k", "share", "window", "seed"))):
    rows = [run(d, keys) for d in sorted((R / sub).glob(pattern)) if (d / "compare.json").exists()]
    json.dump(rows, open(OUT / f"{name}.json", "w"), indent=1)
    print(name, len(rows))
