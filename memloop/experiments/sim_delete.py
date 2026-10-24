import argparse, glob, json, os, pickle, random, sys, zlib

sys.path.insert(0, os.path.expanduser("~/Memloop/memloop"))
from memloop.attribution import responsible_origins, seeds_for_origins

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
        "~/Memloop/results/sim_delete.json"))
    args = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(f"{BASE}/*/seed*")):
        run = "/".join(d.split("/")[-2:])
        for recall in args.recall:
            for fp in args.fp:
                for rep in args.draws:
                    rng = random.Random(zlib.crc32(repr((d, rep, recall, fp)).encode()))
                    store = pickle.load(open(f"{d}/attack.pkl", "rb"))["store"]
                    tr = store.tracker
                    live = [i for i in store.active_ids() if tr.nodes[i].kind == "mem"]
                    att = {i for i in live if tr.steps[tr.nodes[i].step].session == "attacker"}
                    cont = {i for i in live if i not in att and poisoned(tr.nodes[i].meta["record"])}
                    benign = set(live) - att - cont
                    pois = sorted(att | cont)
                    flagged = set(rng.sample(pois, round(recall * len(pois)))) if pois else set()
                    flagged |= {b for b in benign if rng.random() < fp}
                    t = tag(recall, fp, rep)
                    json.dump({i: dict(flag=i in flagged) for i in live},
                              open(f"{d}/detector_flags_sim_{t}.json", "w"))
                    origins = responsible_origins(store, flagged)
                    seeds = set(seeds_for_origins(store, origins))
                    att_steps = {st.trigger for st in tr.steps if st.session == "attacker"}
                    for i in flagged:
                        store.remove(i)
                    pickle.dump(dict(store=store), open(f"{d}/DFs_{t}.pkl", "wb"))
                    rows.append(dict(run=run, recall=recall, fp=fp, draw=rep, tag=t,
                                     live=len(live), poisoned=len(pois), flagged=len(flagged),
                                     kept=len(live) - len(flagged),
                                     attacker_left=len(att - flagged), poisoned_left=len(set(pois) - flagged),
                                     origins=sorted(origins),
                                     oracle_seeds=seeds == att_steps))
                    print(run, t, rows[-1]["kept"], rows[-1]["attacker_left"],
                          rows[-1]["poisoned_left"], rows[-1]["oracle_seeds"], flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
