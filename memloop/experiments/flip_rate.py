import argparse, concurrent.futures as cf, glob, json, os, pickle, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.expanduser("~/Memloop")


def cmd_prompts(a):
    from memloop.adapters.minja_qa import SYSTEM, MinjaQA
    demo = open(f"{ROOT}/baselines/MINJA/QA/initial_demo.txt").read()
    out = []
    for d in sorted(glob.glob(f"{ROOT}/results/qa_recovery/qwen3-14b/*/seed*")):
        store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
        h = MinjaQA(None, store, demo, n_shots=a.shots)
        for st in store.tracker.steps:
            q = store.tracker.nodes[st.trigger].meta["q"]
            hits = h.retrieve(q["question"], st.index)
            out.append(dict(id=f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}/{st.index}",
                            n_examples=len(hits),
                            messages=[{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": h.prompt(q["question"], q["options"], hits)}]))
    random.Random(0).shuffle(out)
    out = out[:a.n]
    with open(a.out, "w") as f:
        for p in out:
            f.write(json.dumps(p) + "\n")
    print(f"{len(out)} prompts -> {a.out}; mean examples per prompt "
          f"{sum(p['n_examples'] for p in out) / len(out):.2f}; mean chars "
          f"{sum(len(p['messages'][1]['content']) for p in out) / len(out):.0f}")


def cmd_run(a):
    from openai import OpenAI
    c = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"), api_key="EMPTY")
    prompts = [json.loads(l) for l in open(a.prompts)]

    def ask(p):
        r = c.chat.completions.create(model=a.model, messages=p["messages"], temperature=0, seed=0,
                                      max_tokens=a.max_tokens, logprobs=True, top_logprobs=2)
        ch = r.choices[0]
        toks = [t.token for t in ch.logprobs.content] if ch.logprobs else []
        margins = [(t.top_logprobs[0].logprob - t.top_logprobs[1].logprob) if len(t.top_logprobs) > 1 else None
                   for t in ch.logprobs.content] if ch.logprobs else []
        return dict(id=p["id"], text=ch.message.content, tokens=toks, margins=margins)

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        res = list(ex.map(ask, prompts))
    json.dump(res, open(a.out, "w"))
    print(f"{len(res)} outputs -> {a.out}")


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def cmd_compare(a):
    runs = {os.path.basename(p)[:-5]: {r["id"]: r for r in json.load(open(p))} for p in a.files}
    names = list(runs)
    report = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = runs[names[i]], runs[names[j]]
            ids = sorted(set(A) & set(B))
            diffs = [first_diff(A[k]["tokens"], B[k]["tokens"]) for k in ids]
            flips = [(k, d) for k, d in zip(ids, diffs) if d is not None]
            margins = [A[k]["margins"][d] for k, d in flips if d < len(A[k]["margins"]) and A[k]["margins"][d] is not None]
            ntok = sum(len(A[k]["tokens"]) for k in ids)
            report[f"{names[i]} vs {names[j]}"] = dict(
                prompts=len(ids), identical=len(ids) - len(flips), flip_rate=len(flips) / len(ids),
                flips_per_1k_tokens=1000 * len(flips) / max(1, ntok),
                median_first_diff_token=sorted(d for _, d in flips)[len(flips) // 2] if flips else None,
                median_top2_margin_at_flip=sorted(margins)[len(margins) // 2] if margins else None)
    json.dump(report, open(a.out, "w"), indent=1)
    for k, v in report.items():
        print(f"{k:40s} identical {v['identical']}/{v['prompts']} (flip rate {v['flip_rate']:.3f}, "
              f"{v['flips_per_1k_tokens']:.2f} per 1k tokens; median first diff at token "
              f"{v['median_first_diff_token']}; median top-2 logprob margin there {v['median_top2_margin_at_flip']})")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prompts"); p.add_argument("--n", type=int, default=600); p.add_argument("--out", required=True)
p.add_argument("--shots", type=int, default=3, help="retrieved examples per prompt (longer prompts)")
p = sub.add_parser("run"); p.add_argument("--prompts", required=True); p.add_argument("--out", required=True)
p.add_argument("--model", default="qwen3-14b"); p.add_argument("--concurrency", type=int, default=32)
p.add_argument("--max-tokens", type=int, default=400)
p = sub.add_parser("compare"); p.add_argument("files", nargs="+"); p.add_argument("--out", required=True)
a = ap.parse_args()
dict(prompts=cmd_prompts, run=cmd_run, compare=cmd_compare)[a.cmd](a)
