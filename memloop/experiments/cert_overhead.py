import glob, json, os, pickle, sys, time
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memloop.ledger import Ledger, canonical, leaf_hash, verify_consistency, verify_inclusion
from memloop.recovery import Certificate, audit_containment, verify_certificate

ROOT = os.path.expanduser("~/Memloop")
KEY = b"memloop-minja"


def timed(f, *a, **k):
    t = time.perf_counter(); r = f(*a, **k); return r, time.perf_counter() - t


def certificate_from_log(led: Ledger) -> Certificate:
    recs = led.records()
    begin_i = max(i for i, r in enumerate(recs) if r["t"] == "recovery_begin")
    b = recs[begin_i]
    run, pre_size, post_size = b["run"], b["pre_size"], len(recs)
    res = [i for i in range(pre_size, post_size) if recs[i]["t"] == "resolve" and recs[i].get("run") == run]
    end = next(r for r in recs[pre_size:] if r["t"] == "recovery_end" and r["run"] == run)
    return Certificate(run=run, mode=b["mode"], seeds=b["seeds"], pre=led.signed_head(pre_size),
                       post=led.signed_head(post_size),
                       consistency=[h.hex() for h in led.consistency_proof(pre_size, post_size)],
                       resolutions=[{"index": i, "record": recs[i],
                                     "proof": [h.hex() for h in led.inclusion_proof(i, post_size)]} for i in res],
                       replayed=[], dropped=[], compensations=[], index_digest=end["index_digest"])


def real():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/results/minja_recovery/qwen3-14b-main/*/seed*")):
        p = f"{d}/recovered_contain.pkl"
        if not os.path.exists(p):
            continue
        store = pickle.load(open(p, "rb"))["store"]
        recs = store.tracker.ledger.records()
        script = json.load(open(f"{d}/script.json"))["script"]
        attack_log = [l for l in open(f"{d}/attack.log") if "] step " in l]
        t0, t1 = attack_log[0][1:9], attack_log[-1][1:9]
        sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8])
        attack_seconds = (sec(t1) - sec(t0)) % 86400 or 1
        led = Ledger(key=KEY)
        _, t_append = timed(lambda: [led.append(r) for r in recs])
        _, t_root = timed(led.root)
        cert, t_cert = timed(certificate_from_log, led)
        cert_bytes = len(json.dumps(cert.__dict__))
        def light():
            ok = verify_consistency(cert.pre["size"], cert.post["size"], bytes.fromhex(cert.pre["root"]),
                                    bytes.fromhex(cert.post["root"]), [bytes.fromhex(h) for h in cert.consistency])
            for r in cert.resolutions:
                ok &= verify_inclusion(leaf_hash(canonical(r["record"])), r["index"], cert.post["size"],
                                       [bytes.fromhex(h) for h in r["proof"]], bytes.fromhex(cert.post["root"]))
            return ok
        ok_light, t_light = timed(light)
        problems, t_full = timed(verify_certificate, cert, recs, KEY)
        cb = next(i for i, r in enumerate(recs) if r["t"] == "contain_begin")
        cprob, t_contain_audit = timed(audit_containment, recs, recs[cb]["pre_size"], recs[cb]["seeds"], recs[cb]["run"])
        pre_attack = recs[:recs[cb]["pre_size"]]
        rows.append(dict(run=d.split("/")[-2] + "/" + d.split("/")[-1], records=len(recs),
                         ledger_bytes=sum(len(canonical(r)) for r in recs),
                         records_per_step=len(pre_attack) / len(script),
                         bytes_per_step=sum(len(canonical(r)) for r in pre_attack) / len(script),
                         append_us_per_record=1e6 * t_append / len(recs),
                         logging_share_of_agent_time=(t_append * len(pre_attack) / len(recs)) / attack_seconds,
                         root_ms_cold=1e3 * t_root, resolutions=len(cert.resolutions),
                         cert_ms=1e3 * t_cert, cert_kb=cert_bytes / 1024,
                         light_verify_ms=1e3 * t_light, light_ok=ok_light,
                         full_audit_ms=1e3 * t_full, full_audit_ok=not problems,
                         containment_audit_ms=1e3 * t_contain_audit, containment_ok=not cprob))
        print(rows[-1]["run"], {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items() if k != "run"},
              flush=True)
    return rows


def synthetic():
    out = []
    for n in (10**3, 10**4, 10**5, 10**6):
        rec = lambda i: {"t": "node", "id": f"mem:{i}", "kind": "mem", "step": i // 3, "pos": i % 3,
                         "parents": [f"obs:{i - 1}"], "trust": 2, "origins": ["user:u"], "c": "ab" * 32, "vec": "cd" * 32}
        led = Ledger(key=KEY)
        _, t_app = timed(lambda: [led.append(rec(i)) for i in range(n)])
        _, t_root = timed(led.root)
        idx = list(range(0, n, max(1, n // 200)))
        proofs, t_prove = timed(lambda: [led.inclusion_proof(i) for i in idx])
        root = led.root()
        ok, t_ver = timed(lambda: all(verify_inclusion(leaf_hash(canonical(led.record(i))), i, n, p, root)
                                      for i, p in zip(idx, proofs)))
        cons, t_cons = timed(led.consistency_proof, n // 2, n)
        out.append(dict(records=n, append_per_s=n / t_app, root_ms_cold=1e3 * t_root,
                        proof_hashes=median(len(p) for p in proofs), proof_bytes=32 * median(len(p) for p in proofs),
                        prove_us=1e6 * t_prove / len(idx), verify_us=1e6 * t_ver / len(idx), verify_ok=ok,
                        consistency_hashes=len(cons), consistency_ms=1e3 * t_cons))
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    res = dict(real=real(), synthetic=synthetic())
    out = f"{ROOT}/results/cert_overhead.json"
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)
