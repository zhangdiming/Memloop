import json, sys
from pathlib import Path
from collections import defaultdict
root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/disp_stress")
rows = []
for c in sorted(root.glob("*/k*/ov*/*/seed*/compare.json")):
    retr, k, ov, win, seed = c.parts[-6:-1]
    cmp = json.load(open(c))
    st = {}
    for tag in ("recovered_contain", "recovered_contain_b3"):
        p = c.parent / f"{tag}_stats.json"
        st[tag] = json.load(open(p)) if p.exists() else {}
    m, b, a = cmp.get("recovered_contain", {}), cmp.get("recovered_contain_b3", {}), cmp.get("attack", {})
    rows.append(dict(retr=retr, k=k[1:], ov=ov[2:], win=win, seed=seed[4:], after=m.get("after_tau"),
                     m_diff=m.get("differ_after_tau"), m_mem=m.get("memory_equal"),
                     b_diff=b.get("differ_after_tau"), b_mem=b.get("memory_equal"), b_jac=b.get("memory_jaccard"),
                     a_diff=a.get("differ_after_tau"),
                     disp=st["recovered_contain"].get("steps_replayed", {}).get("retrieval", 0),
                     checks=st["recovered_contain"].get("displacement_checks"),
                     replayed=sum(st["recovered_contain"].get("steps_replayed", {}).values()),
                     cert=st["recovered_contain"].get("certificate_problems") == [] and
                          st["recovered_contain_b3"].get("certificate_problems") == [],
                     asr=(a.get("test_asr"), m.get("test_asr"), b.get("test_asr"), cmp["counterfactual"]["test_asr"])))
print(f"{'config':32s} after  memloop(diff,memEq)  B3(diff,memEq,jac)  attack_diff  disp/checks  replayed cert  ASR a/m/b/cf")
for r in rows:
    print(f"{r['retr']+' k'+r['k']+' ov'+r['ov']+' '+r['win']+' s'+r['seed']:32s} {r['after']!s:5s} "
          f"{r['m_diff']!s:3s} {r['m_mem']!s:6s}       {r['b_diff']!s:3s} {r['b_mem']!s:6s} {r['b_jac'] or 0:.2f}   "
          f"{r['a_diff']!s:5s}       {r['disp']}/{r['checks']}       {r['replayed']:3d}   {r['cert']}  {r['asr']}")
agg = defaultdict(lambda: [0, 0, 0, 0, 0])
for r in rows:
    for key in (r["retr"], f"{r['retr']} k{r['k']}", f"{r['retr']} ov{r['ov']}", f"{r['retr']} {r['win']}"):
        g = agg[key]; g[0] += 1; g[1] += (r["m_diff"] == 0 and r["m_mem"]); g[2] += (r["b_diff"] == 0 and r["b_mem"])
        g[3] += r["b_diff"] or 0; g[4] += r["after"] or 0
print("\ngroup                 runs  memloop_exact  B3_exact  B3_diff_episodes/after_tau")
for k, g in sorted(agg.items()):
    print(f"{k:22s} {g[0]:4d}  {g[1]:6d}        {g[2]:6d}    {g[3]}/{g[4]} ({g[3]/max(1,g[4]):.1%})")
json.dump(rows, open(root / "summary.json", "w"), indent=1)
