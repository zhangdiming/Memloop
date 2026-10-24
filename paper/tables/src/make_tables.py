import json, math, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TAB = os.path.join(HERE, "..")


def load(name):
    return json.load(open(os.path.join(DATA, f"{name}.json")))


def tab_exact(load):
    det = load("deterministic_pilot")
    pct = lambda x: f"{100 * x:.0f}\\%"
    rows = []
    for pair in ("toothbrush", "jeans", "cookies"):
        c, ev = det[pair]["compare"], det[pair]["eval"]
        rep = "recovered_b3" if pair == "cookies" else "recovered_contain"
        for i, (tag, state) in enumerate((("attack", "attacked"), (rep, "repaired"))):
            rows.append(f"{pair if i == 0 else ''} & {state} & {pct(c[tag]['exact'])} & "
                        f"{c[tag]['memory_jaccard']:.2f} & {pct(ev[tag]['asr'])} & {ev[tag]['memory']} \\\\")
        rows.append("\\addlinespace[2pt]")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Deterministic serving: agreement with the counterfactual (CF) per step and in memory (Jaccard), "
        "residual ASR, and live memory. Repaired: \\sys{} (Taint-Replay identical); for cookies, Taint-Replay, as the "
        "\\sys{} run was lost and a re-run diverged (Section~S3).}",
        "\\label{tab:exact}", "\\footnotesize",
        "\\begin{tabular*}{\\columnwidth}{@{}ll@{\\extracolsep{\\fill}}rrrr@{}}", "\\toprule",
        " & & \\multicolumn{2}{c}{agreement with CF} & & \\\\", "\\cmidrule(lr){3-4}",
        "victim & state & steps & memory & ASR & records \\\\", "\\midrule",
        *rows[:-1], "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])


def tab_dist(load):
    d = load("distributional_pairs")

    def num(x, nd, sign):
        s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
        if float(s) == 0:
            s = f"{0:.{nd}f}"
        elif not sign and x > 0:
            s = "+" + s
        return f"${s}$"

    def cell(key, side, nd):
        s = d[key].get(side)
        if s is None:
            return "--"
        if s["nonzero_runs"] == 0:
            return "0 in every run"
        return f"{num(s['mean'], nd, False)} [{num(s['lo'], nd, True)}, {num(s['hi'], nd, True)}]"

    rows = [f"{label} & {cell(key, 'rec_minus_cf', nd)} & {cell(key, 'cf2_minus_cf', nd)} \\\\"
            for label, key, nd in (("benign success", "benign_success", 3), ("records stored", "stored", 2),
                                   ("target purchases", "target_buys", 0), ("residual ASR", "eval_asr", 0))]
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Default serving, 27 runs: paired per-run differences, mean and 95\\% CI (bootstrap over the nine "
        "victim/target pairs). CF2: second CF run. Benign success: share of benign tasks solved. Target purchases: "
        "non-attacker tasks that bought the attacker's target.}",
        "\\label{tab:dist}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\columnwidth}{@{}l@{\\extracolsep{\\fill}}cc@{}}", "\\toprule",
        "statistic & recovered $-$ CF & CF2 $-$ CF \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])


def tab_s_repro(load):
    rep = load("minja_reproduction")
    pct = lambda x: f"{100 * x:.0f}\\%"
    row = lambda name, isr, asr, inv, cont, ben: f"{name} & {pct(isr)} & {pct(asr)} & {pct(inv)} & {cont}/{ben} \\\\"
    rows = [row(r["pair"].replace("_", " "), r["isr"], r["asr"], r["invalid"], r["benign_with_injected"], r["benign"])
            for r in rep]
    n = len(rep)
    mean = row("mean", *(sum(r[k] for r in rep) / n for k in ("isr", "asr", "invalid")),
               sum(r["benign_with_injected"] for r in rep), sum(r["benign"] for r in rep))
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{MINJA on RAP/WebShop with Qwen3-14B, original settings. ISR: injection success; ASR: attack "
        "success; contaminated: benign records that carry the injected reasoning.}",
        "\\label{tab:s-repro}", "\\footnotesize",
        "\\begin{tabular*}{\\columnwidth}{@{}l@{\\extracolsep{\\fill}}rrrr@{}}", "\\toprule",
        "victim & ISR & ASR & invalid & contaminated \\\\", "\\midrule",
        *rows, "\\midrule", mean, "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])


def tab_s_scale(load):
    ms = load("model_scale")
    models = ("qwen3-8b", "qwen3-14b", "qwen3-32b")
    S = [ms[m]["summary"] for m in models]
    pct = lambda x: f"{100 * x:.0f}\\%"
    line = lambda label, cells: f"{label} & " + " & ".join(cells) + " \\\\"

    def closure_and_memloop(s):
        a, b = pct(s["asr"]["B2"]["mean"]), pct(s["asr"]["recovered_contain"]["mean"])
        assert a == b, (a, b)
        return a

    mem = lambda tag: [f"{s['memory'][tag]['mean']:.1f}" for s in S]
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Model scale on WebShop (3 pairs $\\times$ 2 trials per model, mean over runs). Models are Qwen3.}",
        "\\label{tab:s-scale}", "\\footnotesize",
        "\\begin{tabular*}{\\columnwidth}{@{}l@{\\extracolsep{\\fill}}rrr@{}}", "\\toprule",
        " & 8B & 14B & 32B-AWQ \\\\", "\\midrule",
        line("injection success", [pct(s["inject_success"]["mean"]) for s in S]),
        "\\addlinespace[2pt]", "\\multicolumn{4}{l}{\\emph{residual ASR}} \\\\",
        line("\\quad attacked", [pct(s["asr"]["attack"]["mean"]) for s in S]),
        line("\\quad Delete-Attacker", [pct(s["asr"]["B1"]["mean"]) for s in S]),
        line("\\quad Delete-Closure, \\sys{}", [closure_and_memloop(s) for s in S]),
        "\\addlinespace[2pt]", "\\multicolumn{4}{l}{\\emph{live memory (records)}} \\\\",
        line("\\quad attacked", mem("attack")), line("\\quad Delete-Attacker", mem("B1")),
        line("\\quad Delete-Closure", mem("B2")), line("\\quad \\sys{}", mem("recovered_contain")),
        line("\\quad counterfactual", mem("counterfactual")),
        "\\addlinespace[2pt]", "\\multicolumn{4}{l}{\\emph{repair}} \\\\",
        line("\\quad steps re-run", [pct(s["replay_share"]["mean"]) for s in S]),
        line("\\quad time (min)", [f"{s['minutes']['mean']:.1f}" for s in S]),
        line("\\quad certificates verified", [f"{s['certificates_ok']}/{s['n']}" for s in S]),
        "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])


def tab_listing(load):
    import math
    data = load("listing_runs")
    models = [data[m] for m in ("qwen3-8b", "qwen3-14b", "qwen3-32b")]
    mean = lambda xs: sum(xs) / len(xs)

    def pct(x):
        return "0\\%" if x == 0 else f"{100 * x:.1f}\\%"

    def paired(rs, a, b):
        xs = [r["stored"][a] - r["stored"][b] for r in rs]
        m, n = mean(xs), len(xs)
        ci = 1.96 * math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
        return f"${m:+.1f}\\pm{ci:.1f}$"

    def row(label, cell):
        return f"{label} & " + " & ".join(cell(rs) for rs in models) + r" \\"

    after = lambda rs: max(r["asr"][s] for r in rs for s in ("B1", "B2", "recovered_contain"))
    mem = lambda s: lambda rs: f"{mean([r['memory'][s] for r in rs]):.1f}"
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Malicious-listing attack (3 pairs $\times$ 2 trials per model, default serving, mean over runs unless "
        r"stated; ASR: residual attack success; each recovery: Delete-Attacker, Delete-Closure, and \sys{}). Stored-record "
        r"differences are paired per run (mean $\pm$ 95\% CI); CF2 is a second counterfactual run.}",
        r"\label{tab:listing}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrr@{}}", r"\toprule",
        r" & 8B & 14B & 32B-AWQ \\", r"\midrule",
        row("target purchases, total", lambda rs: str(sum(r["target_purchases"] for r in rs))),
        row("ASR attacked, mean (max)", lambda rs: f"{pct(mean([r['asr']['attack'] for r in rs]))} "
                                                   f"({100 * max(r['asr']['attack'] for r in rs):.0f}\\%)"),
        row("ASR after each recovery", lambda rs: pct(after(rs))),
        r"\addlinespace[2pt]",
        r"\multicolumn{4}{l}{\emph{live memory (records)}} \\",
        row(r"\quad attacked", mem("attack")),
        row(r"\quad Delete-Attacker", mem("B1")),
        row(r"\quad Delete-Closure", mem("B2")),
        row(r"\quad \sys{}", mem("recovered_contain")),
        row(r"\quad counterfactual", mem("counterfactual")),
        r"\addlinespace[2pt]",
        r"\multicolumn{4}{l}{\emph{stored records, paired difference}} \\",
        row(r"\quad \sys{} $-$ CF", lambda rs: paired(rs, "recovered_contain", "counterfactual")),
        row(r"\quad CF2 $-$ CF", lambda rs: paired(rs, "counterfactual2", "counterfactual")),
        r"\addlinespace[2pt]",
        row("steps re-run; repair (min)", lambda rs: f"{100 * sum(r['replayed'] for r in rs) / sum(r['steps_after_tau'] for r in rs):.0f}\\%; "
                                                     f"{mean([r['seconds'] / 60 for r in rs]):.1f}"),
        r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])


def tab_s_qa(load):
    qa = load("qa")
    pct = lambda x: f"{100 * x:.0f}\\%"
    rows = []
    for model, short in (("qwen3-32b", "32B"), ("qwen3-8b", "8B")):
        for r in qa[model]:
            subject, seed = r["run"].split("/")
            if subject != "moral_scenarios" or r["asr"]["attack"] == 0:
                continue
            cells = [f"{pct(r['asr'][t])} & {r['memory'][t]}" for t in ("attack", "B1", "B2", "recovered_contain")]
            rows.append(f"{short}/{seed.removeprefix('seed')} & " + " & ".join(cells)
                        + f" & {r['memory']['counterfactual']} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{QA agent (MMLU, moral scenarios), runs where the attack was effective (run: model/trial). "
        "Residual ASR and live memory records (mem.) per state; CF: counterfactual.}",
        "\\label{tab:s-qa}", "\\footnotesize", "\\setlength{\\tabcolsep}{1.5pt}",
        "\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{attacked} & \\multicolumn{2}{c}{Delete-Attacker} & \\multicolumn{2}{c}{Delete-Closure} "
        "& \\multicolumn{2}{c}{\\sys{}} & CF \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-10}",
        "run & ASR & mem. & ASR & mem. & ASR & mem. & ASR & mem. & mem. \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}", ""])


def tab_s_amem(load):
    pct = lambda x: f"{100 * x:.0f}\\%"
    yes = lambda m: "yes" if m["differ_after_tau"] == 0 and m["memory_equal"] else "no"
    rows = []
    for r in sorted(load("qa_amem"), key=lambda r: (int(r["k"][1:]), int(r["seed"][4:]))):
        a = r["attack"]
        rows.append(f"{r['k'][1:]}/{r['seed'][4:]} & {a['episodes']} & {pct(a['test_asr'])} & {pct(a['exact'])} & "
                    f"{a['memory']} & {a['memory_cf']} & {sum(r['memloop']['replayed'].values())}/{a['after_tau']} & "
                    f"{yes(r['memloop'])} & {yes(r['b3'])} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{QA agent with A-Mem notes (batch-invariant Qwen3-14B). $r$: trial; st.: steps; exact: attacked steps identical to the counterfactual; re-exec.: steps re-executed of those after the attack.}
\label{tab:s-amem}
\footnotesize
\setlength{\tabcolsep}{1pt}
\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrrrcc@{}}
\toprule
 & & \multicolumn{2}{c}{attacked} & \multicolumn{2}{c}{memory} & & \multicolumn{2}{c}{exact recovery} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}
$k$/$r$ & st. & ASR & exact & att. & CF & re-exec. & \sys{} & \shortstack{Taint-\\Replay} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


def tab_s_amem_late(load):
    pct = lambda x: f"{100 * x:.0f}\\%"
    exact = lambda m: m["differ_after_tau"] == 0 and m["memory_equal"]
    runs = sorted(load("qa_amem_late"), key=lambda r: (int(r["k"][1:]), float(r["share"][2:]), int(r["seed"][4:])))
    rows = []
    for r in runs:
        a, m, b = r["attack"], r["memloop"], r["b3"]
        tr = "yes" if exact(b) else f"{b['differ_after_tau']}, {b['memory_jaccard']:.2f}"
        rows.append(f"{r['k'][1:]}/{r['share'][2:]}/{r['seed'][4:]} & {a['episodes']} & {pct(a['test_asr'])} & "
                    f"{pct(a['exact'])} & {a['memory']} & {m['replayed'].get('context', 0)}/{a['after_tau']} & "
                    f"{m['replayed'].get('retrieval', 0)} & {'yes' if exact(m) else 'no'} & {tr} \\\\")
    n = len(runs)
    tot = (f"total & {sum(r['attack']['episodes'] for r in runs)} & & & & "
           f"{sum(r['memloop']['replayed'].get('context', 0) for r in runs)}/{sum(r['attack']['after_tau'] for r in runs)} & "
           f"{sum(r['memloop']['replayed'].get('retrieval', 0) for r in runs)} & "
           f"{sum(exact(r['memloop']) for r in runs)}/{n} & {sum(exact(r['b3']) for r in runs)}/{n} \\\\")
    return r"""\begin{table}[t]
\centering
\caption{QA agent with A-Mem notes, late attack (55--95\% of the schedule; batch-invariant Qwen3-14B). Share: fraction of benign questions from the victim's subject; $r$: trial. Tainted: later benign steps with a tainted input. Displ.: further steps re-executed by the displacement test. Taint-Replay entries give differing steps and final-memory Jaccard similarity.}
\label{tab:s-amem-late}
\footnotesize
\setlength{\tabcolsep}{1pt}
\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrrrcc@{}}
\toprule
 & & \multicolumn{2}{c}{attacked} & & & & \multicolumn{2}{c}{exact recovery} \\
\cmidrule(lr){3-4}\cmidrule(lr){8-9}
$k$/share/$r$ & st. & ASR & exact & mem. & tainted & displ. & \sys{} & \shortstack{Taint-\\Replay} \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + tot + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


def tab_all(load):
    runs = load("full_replay")
    mean = lambda xs: sum(xs) / len(xs)
    rows = [" & ".join([r["run"].split("/")[0].replace("_", " "),
                        *(f"{r[m]['replayed']}/{r[m]['after_tau']}" for m in ("selective", "full")),
                        f"{r['selective']['exact']:.2f}", f"{r['full']['exact']:.2f}", f"{r['cf2_exact']:.2f}",
                        str(r["selective"]["memory"]), str(r["full"]["memory"])]) + " \\\\" for r in runs]
    m = lambda f: mean([f(r) for r in runs])
    rows += [r"\midrule", " & ".join([
        "mean", f"{m(lambda r: r['selective']['replayed']):.0f}", f"{m(lambda r: r['full']['replayed']):.0f}",
        f"{m(lambda r: r['selective']['exact']):.2f}", f"{m(lambda r: r['full']['exact']):.2f}",
        f"{m(lambda r: r['cf2_exact']):.2f}", f"{m(lambda r: r['selective']['memory']):.1f}",
        f"{m(lambda r: r['full']['memory']):.1f}"]) + " \\\\"]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Full replay against selective repair, default serving, trial 0. Steps: re-executed out of those after the first attacker query. Agreement: per-step exact agreement with the counterfactual; CF2 is a second counterfactual run. Residual ASR was 0\% for both.}",
        r"\label{tab:all}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{steps} & \multicolumn{3}{c}{agreement} & \multicolumn{2}{c}{memory} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}",
        r"run & selective & full & sel. & full & CF2 & sel. & full \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]) + "\n"


def tab_s_closed(load):
    det = {r["run"]: r for r in load("detector")["runs"]}
    dd, it = load("detect_delete"), load("iterated_rounds")
    mean = lambda xs: sum(xs) / len(xs)
    pct = lambda v: f"{100 * v:.0f}\\%"
    pct_mean = lambda v: pct(v) if 100 * v >= 10 or v == 0 else f"{100 * v:.1f}\\%"
    cols = {k: [] for k in ("recall", "attacked", "delete", "entry", "origin", "fallback")}
    rows = []
    for run in sorted(det):
        r, s = det[run], it[run]
        named = r["attacker_named"]
        v = dict(recall=r["recall"], attacked=s["attack"]["asr"], delete=dd[run]["DF"]["asr"],
                 entry=s["recovered_contain_entry"]["asr"],
                 origin=s["recovered_contain" if named else "attack"]["asr"],
                 fallback=s["recovered_contain"]["asr"] if named else dd[run]["DF"]["asr"])
        for k in cols:
            cols[k].append(v[k])
        victim, seed = run.split("/")
        rows.append(" & ".join([f"{victim.replace('_', ' ')}/{seed[len('seed'):]}", f"{v['recall']:.2f}",
                                "yes" if named else "no",
                                *(pct(v[k]) for k in ("attacked", "delete", "entry", "origin", "fallback"))]) + " \\\\")
    rows += [r"\midrule", " & ".join([
        "mean", f"{mean(cols['recall']):.2f}", f"{sum(r['attacker_named'] for r in det.values())}/{len(det)}",
        *(pct_mean(mean(cols[k])) for k in ("attacked", "delete", "entry", "origin", "fallback"))]) + " \\\\"]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Closed loop with the LLM detector on all 27 runs: residual ASR by seeding policy. Run: victim/trial; named: attribution named the attacker; delete: Delete-Flagged (the flagged records deleted, nothing re-executed); fallback: origin if an account is named, otherwise delete.}",
        r"\label{tab:s-closed}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rcrrrrr@{}}",
        r"\toprule",
        r"run & recall & named & attacked & delete & entry & origin & fallback \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]) + "\n"


def tab_k2(load):
    runs = load("two_accounts_k2")
    pct = lambda x: f"{100 * x:.0f}\\%"

    def row(r):
        victim, seed = r["run"].split("/")
        unnamed = [a for a in r["accounts"] if a not in r["origins"]]
        inj = str(r["accounts"][unnamed[0]]["injected"]) if len(unnamed) == 1 else "--"
        cells = [inj, pct(r["asr"]["attack"]), pct(r["asr"]["origin"]), str(r["attacker_left"]["origin"]),
                 pct(r["asr"]["union"]), str(r["attacker_left"]["union"]), str(r["cf_memory"])]
        return f"{victim.replace('_', ' ')}/{seed[4:]} & " + " & ".join(cells) + r" \\"

    named = [row(r) for r in runs if r["origins"]]
    none = [row(r) for r in runs if not r["origins"]]
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Attacker spread over two accounts, recovery seeded from the LLM detector's flags. Inj.: the unnamed "
        r"account's queries that injected a record during the attack. Left: attacker records live after recovery.}",
        r"\label{tab:k2}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrrrr@{}}", r"\toprule",
        r" & & & \multicolumn{2}{c}{origin} & \multicolumn{2}{c}{union} & \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"run & inj. & ASR att. & ASR & left & ASR & left & CF mem. \\", r"\midrule",
        *named, r"\addlinespace[2pt]", r"\multicolumn{8}{l}{\emph{attacker not named}} \\", *none,
        r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])


def tab_perquery(load):
    d = load("iterated_rounds")
    mean = lambda xs: sum(xs) / len(xs)
    cf = mean([r["counterfactual"]["memory"] for r in d.values()])

    def row(label, states, benign=True):
        a, left = [s["asr"] for s in states], [s["attacker_live"] for s in states]
        ben = mean([s["live"] - s["poisoned_live"] for s in states]) / cf
        return " & ".join([label, "0\\%" if mean(a) == 0 else f"{100 * mean(a):.1f}\\%", f"{100 * max(a):.0f}\\%",
                           str(sum(x > 0 for x in a)), f"{sum(left)} ({sum(x > 0 for x in left)})",
                           f"{100 * ben:.0f}\\%" if benign else "--"]) + " \\\\"

    rows = []
    for fam, tags in (("entry", ["recovered_contain_entry", "recovered_contain_entry_r2", "recovered_contain_entry_r3"]),
                      ("writer", [f"recovered_contain_writer_r{i}" for i in (1, 2, 3)])):
        prev = {}
        for rnd, t in enumerate(tags, 1):
            cur = {run: v[t] if v[t] is not None else prev[run] for run, v in d.items()}
            rows.append(row(f"{fam}, {rnd}", list(cur.values()), benign=fam == "writer" or rnd == 3))
            prev = cur
    rows += [r"\midrule", row("delete (fallback)", [v["DF"] for v in d.values()]),
             row("origin (one account)", [v["recovered_contain"] for v in d.values()])]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Attacker with one account per query, iterated detect-and-repair rounds (default serving, 27 runs). Entry: seed flagged records. Writer: revoke the query that wrote each flagged record. Delete: the fallback, flagged records deleted once, nothing re-executed. Left: attacker records live (runs with any). Benign: benign records as a share of the counterfactual's memory.}",
        r"\label{tab:perquery}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrr@{}}",
        r"\toprule",
        r"seeds, round & ASR & max & nonzero & left & benign \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]) + "\n"


def tab_recurring(load):
    rows = {(r["detector"], r["k"], r["K"]): r for r in load("attribution_recurring")}
    body = []
    for a in (1, 2):
        if a > 1:
            body.append(r"\addlinespace[2pt]")
        for u in (None, 20, 5, 2):
            cells = []
            for det in ("llm", "sim r=1.0 fp=0.0"):
                r = rows[(det, a, u)]
                cells += [f"{100 * r['att_seeded']:.0f}\\%", f"{100 * r['any_innocent']:.0f}\\%",
                          f"{100 * r['benign_seeded']:.1f}\\%"]
            body.append(f"{a} & {'own' if u is None else u} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Attribution when benign tasks share $u$ recurring accounts and the attacker uses $a$ accounts (27 runs; "
        r"``own'': every task has its own account, as in the main sweep). Att.: attacker sessions seeded. Innoc.: runs in "
        r"which an innocent account was blamed. Benign: benign tasks seeded.}",
        r"\label{tab:recurring}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\columnwidth}{@{}ll@{\extracolsep{\fill}}rrrrrr@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{LLM detector} & \multicolumn{3}{c}{perfect detector} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"$a$ & $u$ & att. & innoc. & benign & att. & innoc. & benign \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])


def tab_s_cert(load):
    import statistics
    real = load("cert_overhead")["real"]
    assert all(r["light_ok"] and r["full_audit_ok"] and r["containment_ok"] for r in real)
    rows = []
    for label, unit, key, scale, nd in (
            ("ledger data per agent step", "KB", "bytes_per_step", 1 / 1024, 2),
            ("append cost per record", "µs", "append_us_per_record", 1, 1),
            ("logging share of agent time", "\\%", "logging_share_of_agent_time", 100, 4),
            ("certificate generation", "ms", "cert_ms", 1, 1),
            ("certificate size", "KB", "cert_kb", 1, 0),
            ("light verification", "ms", "light_verify_ms", 1, 1),
            ("full audit", "ms", "full_audit_ms", 1, 1),
            ("containment audit", "ms", "containment_audit_ms", 1, 2)):
        v = [r[key] * scale for r in real]
        rows.append(f"{label} & {unit} & {statistics.median(v):.{nd}f} & {max(v):.{nd}f} \\\\")
    return "\n".join([
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{Ledger and certificate cost over the {len(real)} main-sweep runs (all certificates verified).}}",
        "\\label{tab:s-cert}", "\\footnotesize",
        "\\begin{tabular*}{\\columnwidth}{@{}l@{\\extracolsep{\\fill}}lrr@{}}", "\\toprule",
        "quantity & unit & median & max \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])


def tab_forget(load):
    runs, rows = load("forget_runs"), []
    for r in runs:
        same = round(r["exact"] * r["episodes"])
        rows.append(f"{r['victim']} & {r['descendants']} & {r['erased_nodes']} & {r['replayed']}/{r['steps_after_tau']} & "
                    f"{r['seconds']:.0f} & {same}/{r['episodes']} \\\\")
    assert all(r["memory_equal"] for r in runs)
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Deletion of one benign record under batch-invariant serving, against a run that never wrote it.}",
        r"\label{tab:forget}", r"\footnotesize",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrr@{}}", r"\toprule",
        r"victim & dependents & erased & re-executed & time (s) & identical \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular*}",
        r"\\[2pt]\parbox{\linewidth}{\scriptsize Re-executed: steps re-run out of the steps after the target. Identical: steps "
        r"identical to the reference; the final memory was identical in all three runs.}",
        r"\end{table}", ""])


def tab_s_evo(load):
    exact = lambda m: m["differ_after_tau"] == 0 and m["memory_equal"]
    mean = lambda xs: f"{sum(xs) / len(xs):.0f}"
    runs = load("qa_amem_evo")
    heads = {"w55": r"\multicolumn{11}{@{}l}{\emph{attack at 55--95\%, 5 templates, 60 benign slots}} \\",
             "w80": r"\multicolumn{11}{@{}l}{\emph{attack at 80--95\%, 2 templates, 120 benign slots}} \\"}

    def _tab_s_evo_tail(g):
        hits = "/".join(str(sum(r["eval"][s]["hits"] for r in g)) for s in ("attack", "B1"))
        disp = [r["memloop"]["replayed"].get("retrieval", 0) for r in g]
        n = len(g)
        return hits, (f"{sum(d > 0 for d in disp)} ({sum(disp)}) & {sum(exact(r['memloop']) for r in g)}/{n} & "
                      f"{sum(exact(r['b3']) for r in g)}/{n} ({sum(r['b3']['differ_after_tau'] for r in g)}) \\\\")

    rows = []
    for w in ("w55", "w80"):
        rows.append(heads[w])
        for k in ("k1", "k3", "k5"):
            for sh in ("ov0.25", "ov1.0"):
                g = [r for r in runs if (r["window"], r["k"], r["share"]) == (w, k, sh)]
                hits, tail = _tab_s_evo_tail(g)
                mem = " & ".join(mean([r["eval"][s]["memory"] for r in g]) for s in ("attack", "B1", "B2", "counterfactual"))
                rows.append(f"{k[1:]}/{sh[2:]} & {sum(r['attack']['after_tau'] for r in g)} & "
                            f"{mean([r['attack']['rewrites'] for r in g])} & {hits} & {mem} & {tail}")
    hits, tail = _tab_s_evo_tail(runs)
    tot = f"total & {sum(r['attack']['after_tau'] for r in runs)} & & {hits} & & & & & {tail}"
    return r"""\begin{table*}[t]
\centering
\caption{QA agent with A-Mem notes and memory evolution (batch-invariant Qwen3-14B; three trials per row). Share: fraction of benign questions from the victim's subject. After: benign steps after the attack. Rewr.: rewrites per run. Hits: held-out attack successes in 30 victim tests, attacked/Delete-Attacker (0 for Delete-Closure, \sys{}, Taint-Replay, and the counterfactual). Mem.: mean live notes; \sys{} equals the counterfactual (CF). Displ.: runs in which the displacement test re-executed further steps (re-executions). Taint-Replay diff.: later steps in which Taint-Replay differs from the counterfactual.}
\label{tab:s-evo}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular*}{0.8\textwidth}{@{}l@{\extracolsep{\fill}}rrrrrrrrcc@{}}
\toprule
 & & & & \multicolumn{4}{c}{mem.} & & \multicolumn{2}{c}{exact runs} \\
\cmidrule(lr){5-8}\cmidrule(lr){10-11}
$k$/share & after & rewr. & hits & att. & Delete-Attacker & Delete-Closure & CF & displ. & \sys{} & \shortstack{Taint-Replay\\(diff.)} \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + tot + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""


def tab_s_during(load):
    runs = load("during_runs")
    body = []
    for i, r in enumerate(runs):
        n = lambda a: sum(t["arrival"] == a for t in r["during"])
        tests = [t for t in r["during"] if t["kind"] == "test"]
        cause = lambda c: sum(x["cause"] == c for x in r["replays"])
        same = round(r["exact"] * r["episodes"])
        first = i == 0 or runs[i - 1]["attack"] != r["attack"] or runs[i - 1]["run_tag"] != r["run_tag"] \
            and r["run_tag"] != "qwen3-14b-inter-shuf"
        if i and runs[i - 1]["run_tag"] == "qwen3-14b-det2" != r["run_tag"]:
            body.append(r"\midrule")
        body.append(f"{r['attack'] if r['victim'] == 'cookies' else ''} & {r['victim']} & "
                    f"{n('before')}/{n('interleaved')}/{n('after')} & {sum(t['hit_target'] for t in tests)}/{len(tests)} & "
                    f"{cause('retrieval')} & {cause('context')} & {same}/{r['episodes']} \\\\")
    assert all(r["memory_jaccard"] == 1.0 for r in runs)
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Six tasks served after containment (batch-invariant Qwen3-14B). Served: tasks served before / during / "
        r"after repair. Displ./context: tasks re-executed by the displacement test or for a recomputed input. Final memory "
        r"equaled the reference in all runs.}",
        r"\label{tab:s-during}", r"\footnotesize",
        r"\begin{tabular*}{\columnwidth}{@{}ll@{\extracolsep{\fill}}rrrrr@{}}", r"\toprule",
        r"attack & victim & served & hits & displ. & context & identical \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])


def tab_s_weak(load):
    det, dd, ms = load("detector"), load("detect_delete"), load("main_sweep")
    sim, bd = load("sim_delete")["runs"], load("blamed_delete")["eval"]
    mean = lambda xs: sum(xs) / len(xs)
    pct = lambda v: "0\\%" if v == 0 else f"{100 * v:.1f}\\%"
    res = lambda asrs: f"{pct(mean(asrs))} ({sum(a > 0 for a in asrs)})"

    def row(label, df, db, named, n, fb):
        return " & ".join([
            label, res([r["asr"] for r in df]), f"{mean([r['attacker_live'] for r in df]):.1f}",
            f"{mean([r['poisoned_live'] for r in df]):.1f}", f"{mean([r['memory'] for r in df]):.1f}",
            f"{pct(db['asr'])} ({db['residual_runs']})", f"{db['poisoned_left']:.1f}", f"{db['memory']:.1f}",
            f"{named}/{n}", res([r["asr"] for r in fb]), f"{mean([r['memory'] for r in fb]):.1f}"]) + " \\\\"

    main = {r["run"]: r for r in ms["runs"]}
    named = {r["run"]: r["attacker_named"] for r in det["runs"]}
    fb = [dict(asr=main[run]["asr"]["recovered_contain"], memory=main[run]["memory"]["recovered_contain"])
          if named[run] else dd[run]["DF"] for run in sorted(dd)]
    rows = [row(f"LLM (recall {det['judge']['recall']:.2f})", [dd[run]["DF"] for run in sorted(dd)], bd["llm"],
                sum(named.values()), len(named), fb)]
    for rc in (0.75, 0.5, 0.25):
        rs = [r for r in sim if r["recall"] == rc]
        rows.append(row(f"recall {rc:.2f}", [r["df"] for r in rs], bd[f"r{round(100 * rc):03d}_f05_d0"],
                        sum(r["memloop_seeds"] == "oracle" for r in rs), len(rs), [r["memloop"] for r in rs]))
    s = ms["summary"]
    note = (f"Attacked memory: {100 * s['asr']['attack']['mean']:.0f}\\% ({s['attack_effective_runs']} runs), "
            f"{s['memory']['attack']['mean']:.1f} records; Delete-Attacker: {100 * s['asr']['B1']['mean']:.1f}\\% "
            f"({s['b1_residual_runs']} runs), {s['memory']['B1']['mean']:.1f} records; counterfactual: "
            f"{s['memory']['counterfactual']['mean']:.1f} records.")
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Responses with the LLM detector and with simulated detectors of lower recall (false-positive rate 5\%, first draw), 27 runs. Delete-Flagged deletes the flagged records. Delete-Blamed also deletes every record written by an account that attribution names. Neither re-executes anything. Residual ASR: mean (runs above 0\%). Att.: attacker records left. Pois.: attacker records and contaminated copies left. Rec.: live records. Named: runs in which attribution named the attacker; in the others, Delete-Blamed and the fallback of \sys{} delete only the flagged records. "
        + note + "}",
        r"\label{tab:s-weak}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}rrrrrrrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Delete-Flagged} & \multicolumn{3}{c}{Delete-Blamed} & \multicolumn{3}{c}{\sys{} (fallback)} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
        r"detector & residual ASR & att. & pois. & rec. & residual ASR & pois. & rec. & named & residual ASR & rec. \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
    ]) + "\n"


def tab_s_alfworld(load):
    d = load("alfworld_main")
    st, pv = d["states"], d["paired_vs_cf"]

    def row(label, tag):
        s, p = st[tag], pv[tag]
        cells = [label, f"{100 * s['asr']:.1f}\\% ({s['residual_runs']})", f"{s['victim_success']:.2f}",
                 "--" if tag == "counterfactual" else
                 f"${p['victim_success']['mean']:+.2f}\\pm{p['victim_success']['ci95']:.2f}$",
                 f"{s['memory']:.1f}",
                 "--" if tag == "counterfactual" else f"${p['memory']['mean']:+.1f}\\pm{p['memory']['ci95']:.1f}$"]
        return " & ".join(cells) + " \\\\"

    main = [("attacked", "attack"), ("Delete-Attacker", "B1"), ("Delete-Closure", "B2"),
            ("Delete-Poisoned", "DP"), ("Taint-Replay", "recovered_b3"), (r"\sys{}", "recovered_contain"),
            ("counterfactual", "counterfactual")]
    sub = [("full replay", "recovered_all"), ("CF2", "counterfactual2")]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RAP on ALFWorld, default serving, 54 runs (trials 0--5); the last two rows cover the 27 runs with trials 0--2. Residual ASR: mean (runs above 0\%). Success: held-out victim tasks solved. Rec.: live records. $\Delta$: paired difference from the counterfactual, mean $\pm$ 95\% CI (1.96 standard errors). Full replay: contain, then re-execute every non-attacker step after the attack.}",
        r"\label{tab:s-alfworld}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrr@{}}",
        r"\toprule",
        r"state & residual ASR & success & $\Delta$ success & rec. & $\Delta$ rec. \\",
        r"\midrule",
        *[row(*x) for x in main],
        r"\midrule",
        *[row(*x) for x in sub],
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]) + "\n"


def tab_s_alfworld_det(load):
    rows = [f"{r['run']} & {r['episodes']} & {100 * r['attack_exact']:.0f}\\% & {r['replayed']}/{r['after_tau']} & "
            f"{r['minutes']:.0f} \\\\" for r in load("alfworld_det")]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RAP on ALFWorld, batch-invariant serving, trial 0: share of non-attacker steps identical to the counterfactual, and re-executed steps out of those after the first attack. \sys{} and Taint-Replay matched in every step with identical final memory in all nine pairs.}",
        r"\label{tab:s-alfworld-det}",
        r"\footnotesize",
        r"\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrr@{}}",
        r"\toprule",
        r"victim & steps & attacked & re-executed & minutes \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]) + "\n"


def _tab_s_utility_ci(diffs, reps=10000, seed=0):
    import random
    by_pair = {}
    for run, v in diffs.items():
        by_pair.setdefault(run.split("/")[0], []).append(v)
    pairs, rng, means = list(by_pair.values()), random.Random(seed), []
    for _ in range(reps):
        s = [v for p in (rng.choice(pairs) for _ in pairs) for v in p]
        means.append(sum(s) / len(s))
    means.sort()
    return sum(diffs.values()) / len(diffs), means[int(0.025 * reps) - 1], means[int(0.975 * reps) - 1]


def tab_s_utility(load):
    runs = load("utility")["runs"]
    named = {r["run"]: r["attacker_named"] for r in load("detector")["runs"]}
    for run, u in runs.items():
        u["loop"] = u["recovered_contain"] if named[run] else u["DF"]
    db = {run: dict(state=v["DB"], cf=v["counterfactual_db"]) for run, v in load("blamed_delete")["utility_runs"].items()}

    def per_run(u, ts, key):
        e = [x for x in u["episodes"] if x["set"] == ts]
        return sum((x["score"] == 1) if key == "success" else x[key] for x in e) / len(e)

    def mean(xs):
        return sum(xs) / len(xs)

    fmt = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"

    def row(label, get):
        pairs = {run: get(run) for run in runs if get(run) is not None}
        cells = [label, f"{mean([s['memory'] for s, _ in pairs.values()]):.1f}"]
        for ts in ("victim", "other"):
            for key in ("success", "score"):
                cells.append(f"{mean([per_run(s, ts, key) for s, _ in pairs.values()]):.3f}")
            if any(c is None for _, c in pairs.values()):
                cells.append("--")
            else:
                m, lo, hi = _tab_s_utility_ci({r: per_run(s, ts, "score") - per_run(c, ts, "score")
                                               for r, (s, c) in pairs.items()})
                cells.append(f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]")
        return " & ".join(cells) + " \\\\"

    same = lambda st: (lambda run: (runs[run][st], runs[run]["counterfactual"]))
    rows = [row("Delete-Attacker", same("B1")), row("Delete-Flagged", same("DF")), row("Delete-Poisoned", same("DP")),
            row("Delete-Closure", same("B2")),
            row("Delete-Blamed$^\\ast$", lambda run: (db[run]["state"], db[run]["cf"]) if run in db else None),
            row("\\sys{}", same("recovered_contain")), row("closed loop", same("loop")),
            row("counterfactual", lambda run: (runs[run]["counterfactual"], None)), row("no memory", same("empty"))]
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Task utility on WebShop, 27 runs, held-out tasks against each final state. Rec.: live records. Success: share of tasks with score 1. $\Delta$ score: paired difference from the counterfactual, mean of per-run differences [95\% CI, bootstrap over the nine victim/target pairs, 10{,}000 resamples]. Closed loop: \sys{} where attribution names the attacker, else Delete-Flagged (Section~\ref{M-sec:eval-closed-loop}). $^\ast$Measured later on new server instances; $\Delta$ against the counterfactual measured on the same instance.}",
        r"\label{tab:s-utility}",
        r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}rrrlrrl@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{20 victim-category tasks} & \multicolumn{3}{c}{10 other tasks} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"state & rec. & success & score & $\Delta$ score & success & score & $\Delta$ score \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
    ]) + "\n"


def tab_s_disp(load):
    exact = lambda m: m["differ_after_tau"] == 0 and m["memory_equal"]
    runs = load("qa_disp_stress")

    def _tab_s_disp_cells(g):
        disp = [r["memloop"]["replayed"].get("retrieval", 0) for r in g]
        n = len(g)
        return (f"{n} & {sum(d > 0 for d in disp)} ({sum(disp)}) & {sum(exact(r['memloop']) for r in g)}/{n} & "
                f"{sum(exact(r['b3']) for r in g)}/{n} & {sum(r['b3']['differ_after_tau'] for r in g)}/"
                f"{sum(r['attack']['after_tau'] for r in g)} \\\\")

    rows = [f"{label} & " + _tab_s_disp_cells([r for r in runs if r["key"] == key])
            for key, label in (("lev", "question, edit distance"), ("dense-q", "question, embedding"))]
    rows.append(r"\multicolumn{6}{@{}l}{question + reasoning, embedding:} \\")
    rows += [f"\\quad $k={k[1:]}$ & " + _tab_s_disp_cells([r for r in runs if (r["key"], r["k"]) == ("dense-qt", k)])
             for k in ("k1", "k3", "k5")]
    return r"""\begin{table}[t]
\centering
\caption{Retrieval-competition stress test on the QA agent (Section~\ref{M-sec:eval-disp}; deterministically served Qwen3-14B, 32 runs). Displ.: runs in which \sys{}'s test triggered a re-execution (re-executions). Exact: runs matching the counterfactual in every step and in final memory. Steps differing: steps after the attack in which Taint-Replay differs from the counterfactual.}
\label{tab:s-disp}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\columnwidth}{@{}l@{\extracolsep{\fill}}rrrrr@{}}
\toprule
 & & & \multicolumn{2}{c}{exact runs} & steps \\
\cmidrule(lr){4-5}
ranking key & runs & displ. & \sys{} & Taint-Replay & differing \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\ntotal & " + _tab_s_disp_cells(runs) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""


TABLES = [tab_exact, tab_dist, tab_s_repro, tab_s_scale, tab_listing, tab_s_qa, tab_s_amem, tab_s_amem_late, tab_all, tab_s_closed, tab_k2, tab_perquery, tab_recurring, tab_s_cert, tab_forget, tab_s_evo, tab_s_during, tab_s_weak, tab_s_alfworld, tab_s_alfworld_det, tab_s_utility, tab_s_disp]


def main():
    check = "--check" in sys.argv[1:]
    bad = 0
    for fn in TABLES:
        path = os.path.join(TAB, f"{fn.__name__}.tex")
        text = fn(load)
        if check:
            same = os.path.exists(path) and open(path).read() == text
            bad += not same
            print(fn.__name__, "ok" if same else "DIFFERS")
        else:
            open(path, "w").write(text)
            print("table", fn.__name__)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
