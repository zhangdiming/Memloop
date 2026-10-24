import glob, json, os, pickle, sys
P = os.path.expanduser("~/Memloop")
sys.path.insert(0, f"{P}/memloop")
sys.path.insert(0, f"{P}/memloop/experiments")
from memloop.attribution import responsible_origins
from attribution_adaptive import relabel

B = f"{P}/results/minja_recovery/qwen3-14b-main"


def attacker_live(store):
    tr = store.tracker
    return sum(tr.nodes[i].kind == "mem" and tr.steps[tr.nodes[i].step].session == "attacker"
               for i in store.active_ids())


out = []
for d in sorted(glob.glob(f"{B}/*/seed*")):
    if not os.path.exists(f"{d}/recovered_contain_origin_k2.pkl"):
        continue
    j = lambda f: json.load(open(f"{d}/{f}"))
    store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
    tr = store.tracker
    relabel(store, 2)
    flagged = [i for i, v in j("detector_flags.json").items() if v["flag"]]
    origins = sorted(responsible_origins(store, flagged))
    acct = {}
    for st in tr.steps:
        if st.session != "attacker":
            continue
        t = tr.nodes[st.trigger]
        a = next(iter(t.label.origins))
        s = acct.setdefault(a, {})
        s[t.meta["session"]] = s.get(t.meta["session"], False) or any(tr.nodes[p].kind == "mem" for p in st.products)
    main = j("eval_attack_B1_B2_recovered_contain_counterfactual.json")
    both = j("eval_recovered_contain_origin_k2_recovered_contain_union_k2.json")
    union = (j("eval_recovered_contain_union_k2.json") if os.path.exists(f"{d}/eval_recovered_contain_union_k2.json")
             else both)["recovered_contain_union_k2"]
    out.append(dict(
        run="/".join(d.split("/")[-2:]), origins=origins,
        accounts={a: dict(sessions=len(s), injected=sum(s.values())) for a, s in sorted(acct.items())},
        asr=dict(attack=main["attack"]["asr"], origin=both["recovered_contain_origin_k2"]["asr"], union=union["asr"]),
        attacker_left={t: attacker_live(pickle.load(open(f"{d}/recovered_contain_{t}_k2.pkl", "rb"))["store"])
                       for t in ("origin", "union")},
        cf_memory=main["counterfactual"]["memory"]))
    print(out[-1]["run"], out[-1]["origins"], out[-1]["accounts"], out[-1]["attacker_left"], file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
