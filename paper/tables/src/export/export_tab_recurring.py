import json, os
rows = json.load(open(os.path.expanduser("~/Memloop/results/attribution_recurring.json")))
print(json.dumps([r for r in rows if r["detector"] in ("llm", "sim r=1.0 fp=0.0")], indent=1))
