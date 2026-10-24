import argparse, glob, json, os, pickle, sys

sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))
from memloop.attribution import responsible_origins, writer_origin

BASE = os.path.expanduser("~/Memloop/results/minja_recovery/qwen3-14b-main")


def poisoned(rec):
    return any("out of stock" in a.lower() for a in rec["Actions"])


def tag(recall, fp, rep):
    return f"r{round(recall * 100):03d}_f{round(fp * 100):02d}_d{rep}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", nargs="+", type=float, default=[0.75, 0.5, 0.25])
    ap.add_argument("--fp", nargs="+", type=float, default=[0.05])
    ap.add_argument("--draws", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default=os.path.expanduser(
        "~/Memloop/results/blamed_delete.json"))
    args = ap.parse_args()
    detectors = [("llm", "detector_flags.json", "DB")]
    detectors += [(t, f"detector_flags_sim_{t}.json", f"DBs_{t}")
                  for t in (tag(r, f, d) for r in args.recall for f in args.fp for d in args.draws)]
    rows = []
    for d in sorted(glob.glob(f"{BASE}/*/seed*")):
        run = "/".join(d.split("/")[-2:])
        for det, flags_file, state in detectors:
            store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
            tr = store.tracker
            live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
            att = {i for i in live if tr.steps[tr.nodes[i].step].session == "attacker"}
            pois = att | {i for i in live if i not in att and poisoned(tr.nodes[i].meta["record"])}
            flags = json.load(open(f"{d}/{flags_file}"))
            flagged = {i for i, v in flags.items() if v["flag"]} & set(live)
            origins = responsible_origins(store, flagged)
            blamed = {i for i in live if writer_origin(tr, i) in set(origins)}
            doomed = blamed | flagged
            for i in doomed:
                store.remove(i)
            pickle.dump(dict(store=store), open(f"{d}/{state}.pkl", "wb"))
            rows.append(dict(run=run, detector=det, state=state, live=len(live), poisoned=len(pois),
                             flagged=len(flagged), origins=sorted(origins),
                             attacker_named=any(tr.steps[tr.nodes[i].step].session == "attacker"
                                                for i in blamed),
                             blamed_records=len(blamed), kept=len(live) - len(doomed),
                             attacker_left=len(att - doomed), poisoned_left=len(pois - doomed)))
            r = rows[-1]
            print(run, state, r["origins"], r["kept"], r["attacker_left"], r["poisoned_left"], flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
