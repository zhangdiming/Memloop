import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import alfworld_recovery as a

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
class A:
    model = "qwen3-14b"; emb_model = "sentence-transformers/all-MiniLM-L6-v2"; num_steps = 50
    pair_idx = 0; seed = 0; inject_num = 5; test_num = 15; eval_num = 20; warmup = 24
warm, benign, *_ = a.schedule(A, a.PAIRS[0])
store = a.new_store()
h = a.harness(A, store, a.PAIRS[0])
def ep(e):
    t0 = time.time()
    o = h.episode(e["gamefile"], e["instruction"], e["category"], "warmup", 0)
    return e, o, time.time() - t0
with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(ep, warm))
for e, o, dt in res:
    print("W", e["category"], e["instruction"], "r=", o.reward, f"{dt:.0f}s", flush=True)
mem = [o.record for _, o, _ in res if o.record]
print("warm-up successes", len(mem), "/", len(res), flush=True)
store = a.new_store(); h.store = store
for e, o, _ in res:
    if o.record:
        ctx = store.begin_step(e["session"])
        from memloop.labels import Trust
        ctx.observe(e["instruction"], "user:w", Trust.USER, trigger=True)
        ctx.emit("mem", json.dumps(o.record), meta={"record": o.record})
for i, e in enumerate(benign[:n]):
    t0 = time.time()
    st = h.run_step(e["session"], "benign", a.taskof(e), f"u{i}")
    o = json.loads(store.tracker.nodes[st.products[-1]].text)
    print(i, e["category"], e["instruction"], "r=", o["reward"], "ret=", len(st.retrieved),
          "mem=", sum(store.tracker.nodes[x].kind == "mem" for x in store.active_ids()),
          f"{time.time()-t0:.0f}s", flush=True)
