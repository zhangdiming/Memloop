import json, os
B = os.path.expanduser("~/Memloop/results/minja_recovery")
out = []
for tag, attack, state in (("qwen3-14b-det2", "shuffled", "recovered_contain_during6"),
                           ("qwen3-14b-inter-shuf", "shuffled", "recovered_contain_during6_inter"),
                           ("qwen3-14b-inter-late", "late", "recovered_contain_during6_inter")):
    for v in ("cookies", "jeans", "toothbrush"):
        d = f"{B}/{tag}/{v}/seed0"
        st = json.load(open(f"{d}/{state}_stats.json"))
        c = json.load(open(f"{d}/compare_counterfactual_during6.json"))[state]
        steps = {t["step"] for t in st["during"]}
        out.append(dict(run_tag=tag, attack=attack, victim=v,
                        during=[dict(step=t["step"], kind=t["kind"], hit_target=t["hit_target"],
                                     arrival=t.get("arrival", "before")) for t in st["during"]],
                        replays=[dict(step=s, cause=cause) for _, s, cause in st["replay_timeline"] if s in steps],
                        episodes=c["episodes"], exact=c["exact"], memory_jaccard=c["memory_jaccard"],
                        audit_ok=not st["certificate_problems"] and not st["containment_problems"]))
print(json.dumps(out, indent=1))
