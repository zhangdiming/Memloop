import json, os
B = os.path.expanduser("~/Memloop/results/minja_recovery/qwen3-14b-det")
out = []
for v in ("cookies", "jeans", "toothbrush"):
    d = f"{B}/{v}/seed0"
    j = lambda f: json.load(open(f"{d}/{f}"))
    tg, st, c = j("forget_target.json"), j("forgotten_stats.json"), j("compare_forget.json")["forgotten"]
    out.append(dict(victim=v, descendants=tg["descendants"], erased_nodes=st["erased_nodes"],
                    replayed=sum(st["steps_replayed"].values()), steps_after_tau=st["steps_after_tau"],
                    seconds=st["seconds"], episodes=c["episodes"], exact=c["exact"],
                    memory_equal=c["memory_equal"],
                    audit_ok=not st["certificate_problems"] and not st["containment_problems"],
                    target_erased=st["target_text_erased"] and st["target_salt_erased"]
                    and not st["target_in_index"] and st["plaintext_in_ledger_records"] == 0))
print(json.dumps(out, indent=1))
