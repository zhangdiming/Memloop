import glob, pickle, sys
sys.path.insert(0, "memloop")
from memloop.recovery import audit, replay_index_digest
import memloop.store

pattern = sys.argv[1] if len(sys.argv) > 1 else "results/minja_recovery/qwen3-14b-main/*/seed*/recovered_contain*.pkl"
bad = n = runs = 0
for path in sorted(glob.glob(pattern)):
    st = pickle.load(open(path, "rb"))["store"]
    recs = st.tracker.ledger.records()
    for r in recs:
        if r["t"] != "recovery_begin":
            continue
        probs = audit(recs, r["pre_size"], r["seeds"], r["run"])
        runs += 1
        if probs:
            bad += 1
            print(path, r["run"], probs[:3])
    if replay_index_digest(recs) != st.index_digest():
        bad += 1
        print(path, "index digest mismatch")
    n += 1
print(f"{n} states, {runs} recovery runs, {bad} with problems")
