import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "..")
os.makedirs(FIG, exist_ok=True)

COL_W, PAGE_W = 3.5, 7.16

FAMILY = "Liberation Sans" if any(f.name == "Liberation Sans" for f in font_manager.fontManager.ttflist) \
    else "DejaVu Sans"

INK, INK2, GRID = "#1a1a1a", "#5a5955", "#e6e5e0"
C = dict(attacked="#cc4b42", B1="#e2a032", DF="#8f5fb0", B2="#1b9e77", memloop="#3266b8", cf="#a3a29d")
NAME = dict(attack="Attacked", B1="Delete-\nAttacker", DF="Delete-\nFlagged", DP="Delete-\nPoisoned",
            B2="Delete-\nClosure", recovered_contain="MemLoop",
            counterfactual="Counter-\nfactual")
COLOR = dict(attack=C["attacked"], B1=C["B1"], DF=C["DF"], DP=C["DF"], B2=C["B2"], recovered_contain=C["memloop"],
             counterfactual=C["cf"])

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.titleweight": "normal", "axes.titlepad": 4,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.bf": f"{FAMILY}:bold",
    "mathtext.cal": FAMILY, "mathtext.it": f"{FAMILY}:italic", "mathtext.sf": FAMILY,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def load(name):
    return json.load(open(os.path.join(DATA, f"{name}.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)
    print("figure", name)


def title(ax, text):
    if text.startswith("("):
        tag, rest = text.split(" ", 1)
        text = r"$\bf{" + tag + "}$ " + rest
    ax.set_title(text, loc="left")


def bars(ax, keys, means, errs, fmt, ylabel, head, runs=None, group=None):
    x = range(len(keys))
    if group:
        lo, hi, label = group
        ax.axvspan(lo - 0.5, hi + 0.5, color="#f3f2ee", lw=0, zorder=0)
    bs = ax.bar(x, means, color=[COLOR[k] for k in keys], width=0.64, edgecolor="white",
                linewidth=0.8, zorder=2)
    for b, k in zip(bs, keys):
        if k == "DP":
            b.set_facecolor("#e8dcef"); b.set_hatch("/////"); b.set_edgecolor(C["DF"])
    top = max(m + e for m, e in zip(means, errs))
    if runs:
        for i, k in enumerate(keys):
            v = runs[k]
            off = [((j * 0.618) % 1 - 0.5) * 0.42 for j in range(len(v))]
            ax.scatter([i + o for o in off], v, s=4, color=INK, alpha=0.38, lw=0, zorder=3)
        top = max(top, max(max(v) for v in runs.values()))
    ax.errorbar(list(x), means, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.2,
                capthick=0.8, zorder=4)
    ax.set_ylim(0, top * 1.17 + 1e-9)
    for i, m in enumerate(means):
        ax.text(i, top * 1.06, fmt(m), ha="center", va="center", fontsize=6.5, color=INK,
                fontweight="bold" if keys[i] == "recovered_contain" else "normal")
    if group:
        ax.text((group[0] + group[1]) / 2, top * 0.96, group[2], ha="center", va="top",
                fontsize=6, color=INK2, style="italic")
    ax.set_xticks(list(x), [NAME[k] for k in keys])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(ylabel)
    title(ax, head)


ms = load("main_sweep")["summary"]
_dd = load("detect_delete")
for _t in ("DF", "DP"):
    for _m, _f in (("asr", "asr"), ("memory", "memory")):
        _v = [r[_t][_f] for r in _dd.values()]
        _mu = sum(_v) / len(_v)
        _sd = (sum((x - _mu) ** 2 for x in _v) / (len(_v) - 1)) ** 0.5
        ms[_m][_t] = dict(mean=_mu, ci=1.96 * _sd / len(_v) ** 0.5, n=len(_v))
keys = ["attack", "B1", "DF", "DP", "B2", "recovered_contain", "counterfactual"]
_runs = load("main_sweep")["runs"]
def per_run(metric, scale):
    out = {}
    for k in keys:
        if k in ("DF", "DP"):
            out[k] = [scale * _dd[r["run"]][k][metric] for r in _runs]
        else:
            out[k] = [scale * r[metric][k] for r in _runs]
    return out
grp = (1, 4, "delete, no replay")
fig, axs = plt.subplots(1, 2, figsize=(PAGE_W, 2.1))
bars(axs[0], keys, [100 * ms["asr"][k]["mean"] for k in keys], [100 * ms["asr"][k]["ci"] for k in keys],
     lambda v: f"{v:.1f}%", "Residual ASR on held-out victims (%)",
     f"(a) Residual attack success ({ms['n']} runs)", per_run("asr", 100), grp)
bars(axs[1], keys, [ms["memory"][k]["mean"] for k in keys], [ms["memory"][k]["ci"] for k in keys],
     lambda v: f"{v:.1f}", "Live memory records", "(b) Memory kept", per_run("memory", 1), grp)
cf = ms["memory"]["counterfactual"]["mean"]
axs[1].axhline(cf, color=C["cf"], lw=0.8, ls=(0, (2, 2)), zorder=1)
fig.tight_layout(w_pad=3)
save(fig, "fig_main_results")

cu = load("recovery_curves")
pairs = ["toothbrush", "jeans", "cookies"]
fig, axs = plt.subplots(1, 3, figsize=(COL_W, 1.75), sharey=True)
ymax = max(max(max(p[1], p[2]) for p in cu[q][v]["series"]) for q in pairs
           for v in ("recover", "contain+repair"))
for ax, pair in zip(axs, pairs):
    rec, con = cu[pair]["recover"], cu[pair]["contain+repair"]
    t = [p[0] for p in rec["series"]]
    ax.fill_between(t, [p[1] for p in rec["series"]], step="post", color=C["attacked"], alpha=0.14, lw=0)
    ax.step(t, [p[1] for p in rec["series"]], where="post", color=C["attacked"], lw=1.1)
    ax.step(t, [p[2] for p in rec["series"]], where="post", color=C["memloop"], lw=1.1)
    tc = [p[0] for p in con["series"]]
    ax.step(tc, [p[1] for p in con["series"]], where="post", color=C["attacked"], lw=1.1,
            ls=(0, (3, 1.5)))
    ax.step(tc, [p[2] for p in con["series"]], where="post", color=C["memloop"], lw=1.1, ls=(0, (3, 1.5)))
    ax.axhline(cu[pair]["counterfactual_memory"], color=INK2, lw=0.6, ls=":")
    ax.set_title(pair, loc="left", fontweight="normal")
    ax.text(0.97, 0.97, f"unsafe\n{rec['unsafe_minutes']:.0f} min", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=C["attacked"])
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_ylim(0, ymax * 1.08)
    ax.tick_params(axis="x", labelsize=6)
axs[0].set_ylabel("live records")
axs[1].set_xlabel("minutes since detection")
handles = [Line2D([], [], color=C["attacked"], lw=1.1, label="contaminated"),
           Line2D([], [], color=C["memloop"], lw=1.1, label="clean"),
           Line2D([], [], color=INK2, lw=1.1, label="replay-only"),
           Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 1.5)), label="contain-then-repair"),
           Line2D([], [], color=INK2, lw=0.6, ls=":", label="counterfactual")]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03),
           handlelength=1.6, columnspacing=0.8, handletextpad=0.4, fontsize=6)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
save(fig, "fig_recovery_curves")

ts = load("attack_time")
wins = ["early", "mid", "late"]
fig, axs = plt.subplots(1, 3, figsize=(PAGE_W, 1.85))
tpairs = sorted({r["pair"] for r in ts})
mk = dict(zip(tpairs, ["o", "s", "^"]))
for i, w in enumerate(wins):
    rs = [r for r in ts if r["window"] == w]
    for ax, f in ((axs[0], lambda r: 100 * r["replayed"] / r["steps"]), (axs[1], lambda r: r["minutes"])):
        for r in rs:
            ax.plot(i, f(r), mk[r["pair"]], color=C["memloop"], ms=4.5, mfc="white", mew=1.0)
        m = sum(f(r) for r in rs) / len(rs)
        ax.hlines(m, i - 0.24, i + 0.24, color=INK, lw=1.2)
        ax.text(i + 0.29, m, f"{m:.0f}", va="center", fontsize=6.5, fontweight="bold")
    for k, dx in (("B2", -0.14), ("recovered_contain", 0.14)):
        vals = [r["memory"][k] / max(1, r["memory"]["counterfactual"]) for r in rs]
        v = 100 * sum(vals) / len(vals)
        axs[2].bar(i + dx, v, width=0.26, color=COLOR[k], edgecolor="white", linewidth=1.0)
        axs[2].text(i + dx, v + 3, f"{v:.0f}", ha="center", fontsize=6.5)
for ax in axs:
    ax.set_xticks(range(3), ["early\n0–20%", "mid\n40–60%", "late\n80–100%"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 2.65)
axs[0].set_ylabel("steps re-run (% of all)"); title(axs[0], "(a) Replay share")
axs[1].set_ylabel("repair time (min)"); title(axs[1], "(b) Repair time")
axs[2].set_ylabel("memory vs. counterfactual (%)"); title(axs[2], "(c) Memory kept")
axs[2].axhline(100, color=INK2, lw=0.6, ls=":")
axs[2].set_ylim(0, 142)
h = [Line2D([], [], marker=mk[p], color=C["memloop"], mfc="white", mew=1.0, ls="", ms=4.5,
            label=p.replace("_", " ")) for p in tpairs]
axs[0].legend(handles=h, loc="upper right", handletextpad=0.2, borderaxespad=0.2)
h2 = [Patch(color=COLOR[k], label=n) for k, n in (("B2", "Delete-Closure"), ("recovered_contain", "MemLoop"))]
axs[2].legend(handles=h2, loc="upper left", ncol=2, handlelength=1.0, handletextpad=0.4,
              borderaxespad=0.2)
fig.tight_layout(w_pad=2.0)
save(fig, "fig_attack_time")

ao = load("attribution_offline")
recalls = [1.0, 0.75, 0.5, 0.25]
fps = (0.0, 0.05, 0.10)
get = lambda key, r, fp, syb=False: next(g[key] for g in ao if g["sybil"] == syb and g["recall"] == r
                                         and g["fp"] == fp)
fig, ax = plt.subplots(figsize=(COL_W, 1.75))
org = [[100 * get("attacker_sessions_seeded", r, fp) for fp in fps] for r in recalls]
ent = [[100 * (1 - get("entry_level_missed_attacker_entries", r, fp)) for fp in fps] for r in recalls]
syb = [100 * get("attacker_sessions_seeded", r, 0.0, True) for r in recalls]
o_mid, e_mid = [v[0] for v in org], [v[0] for v in ent]
ENT_INK = "#a86f12"
ax.fill_between(recalls, e_mid, o_mid, color=C["attacked"], alpha=0.07, lw=0)
for vals, col, m in ((org, C["memloop"], "o"), (ent, C["B1"], "s")):
    ax.fill_between(recalls, [min(v) for v in vals], [max(v) for v in vals], color=col, alpha=0.22, lw=0)
    ax.plot(recalls, [v[0] for v in vals], "-", color=col, lw=1.6, zorder=3, solid_capstyle="round")
    ax.plot(recalls, [v[0] for v in vals], m, color=col, ms=4.2, mec="white", mew=0.8, zorder=4)
ax.plot(recalls, syb, color=C["memloop"], lw=1.0, ls=(0, (3, 2)), zorder=3)
for x, y in zip(recalls[1:], o_mid[1:]):
    ax.text(x, y + 4, f"{y:.0f}", ha="center", va="bottom", fontsize=6, color=C["memloop"])
for x, y in zip(recalls[1:], e_mid[1:]):
    ax.text(x, y - 6, f"{y:.0f}", ha="center", va="top", fontsize=6, color=ENT_INK)
xg = 0.31
yo, ye = 96.3 + (92.6 - 96.3) * (0.5 - xg) / 0.25, 47.8 + (23.5 - 47.8) * (0.5 - xg) / 0.25
ax.annotate("", xy=(xg, ye + 2), xytext=(xg, yo - 2),
            arrowprops=dict(arrowstyle="<->", color=C["attacked"], lw=0.8, shrinkA=0, shrinkB=0))
ax.text(xg + 0.015, (yo + ye) / 2, "left in\nmemory by\nentry seeds", color=C["attacked"],
        fontsize=6, ha="right", va="center", style="italic", linespacing=1.1)
ax.text(0.95, 102, "origin-level", color=C["memloop"], fontsize=6.5, ha="left", fontweight="bold")
ax.text(0.84, 52, "entry-level", color=ENT_INK, fontsize=6.5, ha="left", fontweight="bold")
ax.text(0.97, 5, "origin-level, fresh account per query: none named", color=C["memloop"],
        fontsize=6, ha="left", va="bottom")
ax.invert_xaxis()
ax.set_xticks(recalls, ["1.0", ".75", ".5", ".25"])
ax.set_xlim(1.04, 0.21)
ax.set_ylim(-5, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.spines["bottom"].set_bounds(1.0, 0.25)
ax.spines["left"].set_bounds(0, 100)
ax.set_xlabel("detector recall (weaker \u2192)")
ax.set_ylabel("attacker records removed (%)")
fig.tight_layout()
save(fig, "fig_attribution")

pa = load("parallel")["widths"]
fig, ax = plt.subplots(figsize=(COL_W, 1.6))
ys = range(len(pa))[::-1]
ax.barh(list(ys), [p["minutes"] for p in pa], color=C["memloop"], height=0.58, edgecolor="white")
for y, p in zip(ys, pa):
    ax.text(p["minutes"] + 1.5, y, f"{p['minutes']:.1f} min   {pa[0]['minutes'] / p['minutes']:.2f}×",
            va="center", fontsize=6.5, fontweight="bold")
    if p["width"] > 1:
        ax.text(2, y, f"{100 * p['conflicts'] / p['replays']:.0f}% conflicts", va="center",
                fontsize=6, color="white")
ax.set_yticks(list(ys), [f"width {p['width']}" for p in pa])
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.5); ax.grid(axis="y", visible=False)
ax.set_xlabel("repair wall-clock time (min)")
ax.set_xlim(0, max(p["minutes"] for p in pa) * 1.42)
title(ax, "Parallel repair (final state identical across widths)")
fig.tight_layout()
save(fig, "fig_parallel")
