import concurrent.futures as cf, json, pickle, sys
from openai import OpenAI
out_path, attack_pkl = sys.argv[1], sys.argv[2]
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
recs = [n.meta["record"] for n in pickle.load(open(attack_pkl, "rb"))["store"].tracker.nodes.values()
        if n.kind == "mem"]
def minja_style(i):
    ex = "\n\n".join(f"This is example {j+1}:\nInstruction: {r['Instruction']}\n" + "\n".join(r["Actions"])
                     for j, r in enumerate(recs[i:i + 3]))
    return ("Interact with a webshop application. Here are examples.\n" + ex +
            f"\n\nHere is the task.\nInstruction: {recs[i + 5]['Instruction']}\nAction:")
prompts = ["Reply with exactly: search[electric toothbrush]",
           "You are shopping on WebShop. Instruction: i need a pack of 2 sugar free cookies under 20 "
           "dollars. Think step by step about which search query to use, then write the query."]
prompts += [minja_style(i) for i in range(0, 30, 5)]
def ask(p):
    r = c.chat.completions.create(model="qwen3-14b", temperature=0, seed=0, max_tokens=200,
                                  messages=[{"role": "system", "content": "You are a helpful assistant for household task."},
                                            {"role": "user", "content": p}])
    return r.choices[0].message.content
alone = [ask(p) for p in prompts]
with cf.ThreadPoolExecutor(len(prompts)) as ex:
    together = list(ex.map(ask, prompts))
json.dump(dict(alone=alone, together=together), open(out_path, "w"), indent=1)
print(f"{len(prompts)} prompts; alone==together within this server: {sum(a == b for a, b in zip(alone, together))}/{len(prompts)}")
