"""
Figure and table generation for the AIH manuscript.

Reads the result CSVs produced by the experiment suite and writes
publication-ready figures (600 dpi PNG + editable SVG) and table CSVs.

Usage from a terminal:
    python make_figures.py                        # expects CSVs in ./results
    python make_figures.py --csv /path/dir --out /path/figs

Usage in Colab or Jupyter, any of these work:
    !python make_figures.py --csv /content/results --out /content/figures
    %run make_figures.py --csv /content/results --out /content/figures
    import os; os.environ["DF_CSV"] = "/content/results"
    os.environ["DF_OUT"] = "/content/figures"; exec(open("make_figures.py").read())

Figures 3 and 7 recompute depth curves from the source catalogue, which is
downloaded automatically (no credentials required).
"""

import os, re, io, sys, json, tarfile, urllib.request, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------------------
# style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# colour-blind safe, prints legibly in greyscale
C_RULE     = "#0F6E56"   # teal
C_LEARNED  = "#993C1D"   # coral
C_HYBRID   = "#185FA5"   # blue
C_ORACLE   = "#5F5E5A"   # grey
C_BASE     = "#888780"
C_ACCENT   = "#854F0B"   # amber
PALETTE    = [C_RULE, C_LEARNED, C_HYBRID, C_ACCENT, C_ORACLE, C_BASE]

# Notebook-safe argument parsing. Jupyter and Colab inject their own -f kernel.json
# argument into sys.argv, so parse_known_args is used and unknown flags are ignored.
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.environ.get("DF_CSV", "./results"))
ap.add_argument("--out", default=os.environ.get("DF_OUT", "./figures"))
args, _unknown = ap.parse_known_args()
CSV, OUT = args.csv, args.out
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(CSV, name)
    if not os.path.exists(p):
        print(f"  [skip] {name} not found in {CSV}")
        return None
    return pd.read_csv(p)


def save(fig, stem):
    fig.savefig(os.path.join(OUT, f"{stem}.png"))
    fig.savefig(os.path.join(OUT, f"{stem}.svg"))
    plt.close(fig)
    print(f"  wrote {stem}.png / .svg")


def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    bs = v[g.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ============================================================================
# Figure 1 - observability decision framework (schematic)
# ============================================================================
def fig1_framework():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)

    rows = [
        ("Fully observable",     "Hazard exact at inference",
         "Transparent analytical rule",  "Provably optimal; learning ties",   C_RULE,    4.4),
        ("Partially observable", "Declaration incomplete",
         "Hybrid lookup or estimate",    "First strategy to win",             C_ACCENT,  2.6),
        ("Hidden",               "Hazard not recoverable",
         "Learned hazard estimator",     "Learning significantly better",     C_LEARNED, 0.8),
    ]
    for cond, cond_sub, strat, strat_sub, col, y in rows:
        ax.add_patch(FancyBboxPatch((0.25, y), 3.7, 1.35, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor="white", edgecolor=col, linewidth=1.1))
        ax.text(2.10, y + 0.86, cond, ha="center", va="center", fontsize=9.5, color=col, weight="bold")
        ax.text(2.10, y + 0.44, cond_sub, ha="center", va="center", fontsize=8, color="#444441")

        ax.add_patch(FancyBboxPatch((5.55, y), 4.2, 1.35, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor=col, edgecolor=col, linewidth=1.1, alpha=0.10))
        ax.add_patch(FancyBboxPatch((5.55, y), 4.2, 1.35, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor="none", edgecolor=col, linewidth=1.1))
        ax.text(7.65, y + 0.86, strat, ha="center", va="center", fontsize=9.5, color=col, weight="bold")
        ax.text(7.65, y + 0.44, strat_sub, ha="center", va="center", fontsize=8, color="#444441")

        ax.add_patch(FancyArrowPatch((4.05, y + 0.68), (5.45, y + 0.68),
                                     arrowstyle="-|>", mutation_scale=11, color=col, linewidth=1.0))

    for y0, y1 in [(4.4, 3.95), (2.6, 2.15)]:
        ax.add_patch(FancyArrowPatch((2.10, y0), (2.10, y1), arrowstyle="-|>",
                                     mutation_scale=11, color="#888780", linewidth=0.9))
    ax.text(0.25, 0.15, "Shading marks decreasing hazard observability",
            fontsize=7.5, color="#5F5E5A")
    save(fig, "fig1_framework")


# ============================================================================
# Figure 2 - core pipeline (schematic)
# ============================================================================
def fig2_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.4); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, sub, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                    facecolor="white", edgecolor=col, linewidth=1.0))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8.8,
                color=col, weight="bold")
        ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center", fontsize=7.2, color="#444441")

    box(0.15, 1.55, 2.3, 1.20, "Declared attributes", "Tokenized item text", C_BASE)
    box(3.00, 2.85, 2.6, 1.20, "Reference lists", "Three severity tiers", C_RULE)
    box(3.00, 0.25, 2.6, 1.20, "Segment labels", "Five user segments", C_HYBRID)
    box(6.15, 2.85, 2.4, 1.20, "Hazard score R", "Deterministic lookup", C_RULE)
    box(6.15, 0.25, 2.4, 1.20, "Relevance model", "Out-of-fold scores", C_HYBRID)
    box(9.10, 1.55, 2.7, 1.20, "Re-ranking rule", "rel minus scaled hazard", C_ACCENT)

    for a, b, col in [((2.50, 2.35), (2.95, 3.20), C_RULE),
                      ((2.50, 1.95), (2.95, 1.10), C_HYBRID),
                      ((5.65, 3.45), (6.10, 3.45), C_RULE),
                      ((5.65, 0.85), (6.10, 0.85), C_HYBRID),
                      ((8.60, 3.20), (9.05, 2.40), C_ACCENT),
                      ((8.60, 0.95), (9.05, 1.85), C_ACCENT)]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                     color=col, linewidth=0.9))
    save(fig, "fig2_pipeline")


# ============================================================================
# Figure 3 - observability protocol (schematic)
# ============================================================================
def fig3_protocol():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 12); ax.set_ylim(-0.3, 7.6); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, sub, col, fill=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                    facecolor=col if fill else "white",
                                    alpha=0.12 if fill else 1.0, edgecolor=col, linewidth=1.0))
        if fill:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                        facecolor="none", edgecolor=col, linewidth=1.0))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=8.4,
                color=col, weight="bold")
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center", fontsize=7.0, color="#444441")

    def arrow(a, b, col, style="-|>"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=10,
                                     color=col, linewidth=0.9))

    # ground truth
    box(4.55, 6.35, 3.0, 1.05, "True hazard", "Ground truth for scoring", C_ORACLE)

    # two corruption modes
    box(0.90, 4.75, 3.1, 1.05, "Masking at rate p", "Tokens undeclared", C_LEARNED)
    box(8.10, 4.75, 3.1, 1.05, "Value noise at scale \u03b5", "Values corrupted", C_ACCENT)
    arrow((5.30, 6.30), (2.60, 5.85), C_LEARNED)
    arrow((6.80, 6.30), (9.50, 5.85), C_ACCENT)

    # what each corruption leaves behind
    box(0.90, 3.15, 3.1, 1.05, "Lookup and text lost", "Both sources damaged", C_LEARNED, fill=True)
    box(8.10, 3.15, 3.1, 1.05, "Lookup lost, text kept", "Evidence survives", C_ACCENT, fill=True)
    arrow((2.45, 4.70), (2.45, 4.25), C_LEARNED)
    arrow((9.65, 4.70), (9.65, 4.25), C_ACCENT)

    # audited subset feeding the estimator
    box(4.40, 4.75, 3.3, 1.05, "Audited subset", "30 percent verified", C_HYBRID)
    box(4.40, 3.15, 3.3, 1.05, "Learned estimator", "Predicts hazard from text", C_HYBRID)
    arrow((6.05, 6.30), (6.05, 5.85), C_HYBRID)
    arrow((6.05, 4.70), (6.05, 4.25), C_HYBRID)

    # the four systems
    labels = [("Oracle", "true hazard", C_ORACLE), ("Rule", "observed hazard", C_RULE),
              ("Learned", "estimated hazard", C_LEARNED), ("Hybrid", "larger of the two", C_HYBRID)]
    for i, (t, sub, col) in enumerate(labels):
        x = 0.35 + i * 2.95
        box(x, 1.45, 2.65, 1.05, t, sub, col)
        arrow((x + 1.32, 3.10), (x + 1.32, 2.55), "#888780")

    # collector line then a single arrow into evaluation
    ax.plot([1.67, 10.52], [1.20, 1.20], color="#888780", linewidth=0.9)
    for i in range(4):
        x = 0.35 + i * 2.95 + 1.32
        ax.plot([x, x], [1.40, 1.20], color="#888780", linewidth=0.9)
    arrow((6.05, 1.20), (6.05, 0.92), "#888780")
    box(3.10, -0.15, 5.9, 0.95, "Evaluation against true hazard",
        "Matched ranking quality", C_ACCENT)
    save(fig, "fig3_protocol")

# ============================================================================
# shared: catalogue loader for recomputed depth curves
# ============================================================================
EU26 = set(["amyl cinnamal","amylcinnamyl alcohol","benzyl alcohol","benzyl salicylate",
"cinnamyl alcohol","cinnamal","citral","coumarin","eugenol","geraniol","hydroxycitronellal",
"hydroxyisohexyl 3-cyclohexene carboxaldehyde","isoeugenol","anisyl alcohol","benzyl benzoate",
"benzyl cinnamate","citronellol","farnesol","hexyl cinnamal","lilial","d-limonene","limonene",
"linalool","methyl 2-octynoate","alpha-isomethyl ionone","evernia prunastri","evernia furfuracea"])
L2A = set(["retinol","retinyl palmitate","retinaldehyde","retinyl retinoate","glycolic acid",
"lactic acid","salicylic acid","mandelic acid","ascorbic acid","benzoyl peroxide","azelaic acid"])
SEGMENTS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]


def _tok(s):
    out = []
    for p in re.split(r"[,;]", str(s).lower()):
        p = re.sub(r"\(.*?\)", "", p)
        p = re.sub(r"[^a-z0-9\- ]", " ", p).strip()
        p = re.sub(r"\s+", " ", p)
        if p:
            out.append(p)
    return out


def catalogue():
    last = None
    for br in ("main", "master"):
        try:
            url = ("https://codeload.github.com/farahjbara/"
                   f"Comparing-Cosmetics-by-Ingredients/tar.gz/refs/heads/{br}")
            raw = urllib.request.urlopen(url, timeout=60).read()
            with tarfile.open(fileobj=io.BytesIO(raw)) as t:
                m = [x for x in t.getmembers() if x.name.endswith("cosmetics.csv")][0]
                df = pd.read_csv(t.extractfile(m))
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"catalogue download failed: {last}")

    df["tok"] = df["Ingredients"].apply(_tok)
    df = df[df["tok"].map(len) >= 3].reset_index(drop=True)
    frag = lambda t: ("fragrance" in t) or ("parfum" in t) or (t == "perfume")
    l1 = df["tok"].apply(lambda tl: sum(x in EU26 for x in tl)).values.astype(float)
    l2 = df["tok"].apply(lambda tl: sum(x in L2A for x in tl)).values.astype(float)
    l3 = df["tok"].apply(lambda tl: sum(frag(x) for x in tl)).values.astype(float)
    return df, l1, l2, l3, 3 * l1 + 2 * l2 + l3


def depth_curve(scores, R, kmax=200):
    o = np.argsort(-scores, kind="mergesort")
    Rs = R[o]
    k = np.arange(1, len(Rs) + 1)
    F = np.cumsum(Rs) / k
    return k[:kmax], F[:kmax]


# ============================================================================
# Figure 4 - depth drift curves and index by segment
# ============================================================================
def fig3_depth():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.naive_bayes import MultinomialNB

    df, l1, l2, l3, R = catalogue()
    X = TfidfVectorizer(token_pattern=r"[^ ]+", min_df=2).fit_transform(
        df["tok"].apply(lambda t: " ".join(t)))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for i, seg in enumerate(SEGMENTS):
        y = df[seg].values
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            oof[te] = MultinomialNB().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        k, F = depth_curve(oof, R, kmax=200)
        ax[0].plot(k, F, color=PALETTE[i % len(PALETTE)], linewidth=1.2, label=seg)
    ax[0].set_xlabel("Recommendation depth k")
    ax[0].set_ylabel("Mean hazard of top k")
    ax[0].legend(frameon=False, ncol=2, fontsize=7)

    C = load("C_dfi_sensitivity.csv")
    if C is not None:
        frac = C.groupby("skin")["DFI"].apply(lambda v: (v > 0).mean()).reindex(SEGMENTS)
        bars = ax[1].bar(range(len(frac)), frac.values, color=C_RULE, alpha=0.85, width=0.62)
        ax[1].axhline(0.5, color=C_LEARNED, linewidth=0.9, linestyle="--")
        ax[1].set_xticks(range(len(frac)))
        ax[1].set_xticklabels(frac.index, rotation=30, ha="right")
        ax[1].set_ylabel("Fraction of settings with positive index")
        ax[1].set_ylim(0, 1.05)
        for b, v in zip(bars, frac.values):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                       ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig4_depth_drift")


# ============================================================================
# Figure 5 - mitigation battery, both pools
# ============================================================================
def fig4_battery():
    D = load("D_battery.csv")
    if D is None:
        return
    order = ["relevance_only", "hard_filter", "FAIR_prefix", "MMR", "proportional", "proposed"]
    labels = ["Relevance only", "Removal", "Prefix constraint", "Diversity", "Proportional", "Proposed"]
    ks = [5, 10, 20, 50]
    pools = [p for p in ["dense", "full"] if p in set(D["pool"])]

    fig, axes = plt.subplots(len(pools), 2, figsize=(7.2, 2.9 * len(pools)), squeeze=False)
    for r, pool in enumerate(pools):
        sub = D[D["pool"] == pool]
        for m, lab, col in zip(order, labels, PALETTE + PALETTE):
            if m not in set(sub.method):
                continue
            g = sub[sub.method == m]
            axes[r][0].plot(ks, [g[f"burden@{k}"].mean() for k in ks], marker="o",
                            markersize=3.5, linewidth=1.1, color=col, label=lab)
            axes[r][1].plot(ks, [g[f"ndcg@{k}"].mean() for k in ks], marker="o",
                            markersize=3.5, linewidth=1.1, color=col, label=lab)
        axes[r][0].set_ylabel(f"{pool} pool\nMean hazard at k")
        axes[r][1].set_ylabel("nDCG at k")
        for c in (0, 1):
            axes[r][c].set_xlabel("k")
            axes[r][c].set_xticks(ks)
    axes[0][1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "fig5_battery")

    rows = []
    for pool in pools:
        sub = D[D["pool"] == pool]
        for m, lab in zip(order, labels):
            if m not in set(sub.method):
                continue
            g = sub[sub.method == m]
            rows.append(dict(Pool=pool, Method=lab,
                             **{f"Hazard@{k}": round(g[f"burden@{k}"].mean(), 3) for k in [10, 50]},
                             **{f"nDCG@{k}": round(g[f"ndcg@{k}"].mean(), 3) for k in [10, 50]}))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table5_battery.csv"), index=False)
    print("  wrote table5_battery.csv")


# ============================================================================
# Figure 6 - learned mitigation steelman
# ============================================================================
def fig5_steelman():
    E = load("E_steelman.csv")
    if E is None:
        return
    meths = ["rule", "C_nestedCV", "B_lambdamart", "A_text_objective", "D_attr_as_feature"]
    labels = ["Analytical rule", "Learned weights", "LambdaMART", "Text objective", "Attribute as feature"]
    floors = [0.95, 0.90, 0.85, 0.80, 0.75]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for m, lab, col in zip(meths, labels, PALETTE):
        vals = []
        for fl in floors:
            v = E[(E.method == m) & (E.ndcg >= fl)]
            vals.append(v.burden.min() if len(v) else np.nan)
        ax[0].plot(floors, vals, marker="o", markersize=3.5, linewidth=1.1, color=col, label=lab)
    ax[0].set_xlabel("nDCG floor"); ax[0].set_ylabel("Best achievable hazard at 10")
    ax[0].invert_xaxis(); ax[0].legend(frameon=False, fontsize=7)

    keys = ["skin", "seed", "fold"]
    names, means, los, his = [], [], [], []
    for m, lab in zip(meths[1:], labels[1:]):
        d = []
        for _, g in E.groupby(keys):
            r = g[(g.method == "rule") & (g.ndcg >= 0.80)].burden.min()
            c = g[(g.method == m) & (g.ndcg >= 0.80)].burden.min()
            if np.isfinite(r) and np.isfinite(c):
                d.append(c - r)
        if len(d) > 1:
            lo, hi = boot_ci(d)
            names.append(lab); means.append(np.mean(d)); los.append(lo); his.append(hi)
    ypos = np.arange(len(names))
    ax[1].errorbar(means, ypos, xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                   fmt="o", color=C_LEARNED, ecolor=C_BASE, capsize=3, markersize=4, linewidth=1.0)
    ax[1].axvline(0, color=C_RULE, linewidth=1.0)
    ax[1].set_yticks(ypos); ax[1].set_yticklabels(names)
    ax[1].set_xlabel("Hazard difference from analytical rule")
    fig.tight_layout()
    save(fig, "fig6_steelman")

    pd.DataFrame(dict(Method=names, Mean=np.round(means, 3),
                      CI_low=np.round(los, 3), CI_high=np.round(his, 3))).to_csv(
        os.path.join(OUT, "table8_steelman.csv"), index=False)
    print("  wrote table8_steelman.csv")


# ============================================================================
# Figure 7 - observability study
# ============================================================================
def fig6_observability():
    F = load("F_observability.csv")
    if F is None:
        return
    LAM = 0.6                      # fixed-policy comparison: same rule, hazard input differs
    F = F[np.isclose(F["lam"], LAM)]
    front = lambda s: (s["burden"].mean() if len(s) else np.nan)
    modes = list(F["mode"].unique())
    sysmap = [("oracle", "Oracle", C_ORACLE, ":"), ("rule", "Analytical rule", C_RULE, "-"),
              ("learned", "Learned estimator", C_LEARNED, "-"), ("hybrid", "Hybrid", C_HYBRID, "--")]

    fig, axes = plt.subplots(1, len(modes), figsize=(3.7 * len(modes), 2.9), squeeze=False)
    rows = []
    for a, mode in zip(axes[0], modes):
        sm = F[F["mode"] == mode]
        lv = sorted(sm.level.unique())
        for sy, lab, col, ls in sysmap:
            a.plot(lv, [front(sm[(sm.system == sy) & (sm.level == l)]) for l in lv],
                   marker="o", markersize=3.5, linewidth=1.1, color=col, linestyle=ls, label=lab)
        a.set_xlabel("Masking rate" if mode == "mask" else "Value noise")
        a.set_ylabel("Mean hazard at 10")
        for l in lv:
            d = []
            for sd in sm.seed.unique():
                for st in sm.skin.unique():
                    q = sm[(sm.level == l) & (sm.seed == sd) & (sm.skin == st)]
                    r, c = front(q[q.system == "rule"]), front(q[q.system == "learned"])
                    if np.isfinite(r) and np.isfinite(c):
                        d.append(r - c)
            if len(d) > 1:
                lo, hi = boot_ci(d)
                rows.append(dict(Mode=mode, Level=l, Mean=round(np.mean(d), 3),
                                 CI_low=round(lo, 3), CI_high=round(hi, 3),
                                 n=len(d), Significant="yes" if lo > 0 else "no"))
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig7_observability")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table10_observability.csv"), index=False)
    print("  wrote table10_observability.csv")


# ============================================================================
# Figure 8 - conditionality across hazard definitions
# ============================================================================
def fig7_conditionality():
    H = load("H_domains.csv")
    if H is None:
        return
    H = H.copy()
    H["gain"] = H["burden10_rel"] - H["burden10_rule"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    doms = list(H.domain.unique())
    w = 0.38
    for i, d in enumerate(doms):
        g = H[H.domain == d].set_index("segment").reindex(SEGMENTS)
        ax[0].bar(np.arange(len(SEGMENTS)) + (i - 0.5) * w, g["DFI"].values, width=w,
                  color=PALETTE[i], alpha=0.85,
                  label="Fragrance allergen list" if i == 0 else "Chemical concern list")
    ax[0].axhline(0, color="#444441", linewidth=0.8)
    ax[0].set_xticks(range(len(SEGMENTS)))
    ax[0].set_xticklabels(SEGMENTS, rotation=30, ha="right")
    ax[0].set_ylabel("Depth index")
    ax[0].legend(frameon=False, fontsize=7)

    for i, d in enumerate(doms):
        g = H[H.domain == d]
        ax[1].scatter(g["DFI"], g["gain"], s=26, color=PALETTE[i], alpha=0.9,
                      label="Fragrance allergen list" if i == 0 else "Chemical concern list")
    ax[1].set_xlabel("Depth index"); ax[1].set_ylabel("Hazard reduction from mitigation")
    ax[1].axvline(0, color="#444441", linewidth=0.8)
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig8_conditionality")


# ============================================================================
# Table 1 - dataset and hazard tiers
# ============================================================================
def table1_data():
    df, l1, l2, l3, R = catalogue()
    rows = [
        dict(Property="Items after filtering", Value=len(df)),
        dict(Property="Items carrying a tier 1 token", Value=int((l1 > 0).sum())),
        dict(Property="Mean hazard score", Value=round(float(R.mean()), 2)),
        dict(Property="Items with zero hazard", Value=int((R == 0).sum())),
        dict(Property="Tier 1 vocabulary size", Value=len(EU26)),
        dict(Property="Tier 2 vocabulary size", Value=len(L2A)),
        dict(Property="Segment labels", Value=len(SEGMENTS)),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table1_data.csv"), index=False)
    print("  wrote table1_data.csv")



# ============================================================================
# Table 3 - learned mitigation formulations
# ============================================================================
def table3_formulations():
    rows = [
        dict(Formulation="A", Inputs="Text features",
             Objective="Pointwise relevance plus browsing-weighted hazard gradient",
             Hazard="Estimated from tokens"),
        dict(Formulation="B", Inputs="Text features",
             Objective="LambdaMART on hazard-discounted graded labels",
             Hazard="Estimated from tokens"),
        dict(Formulation="C", Inputs="Three hazard tier counts",
             Objective="Smoothed browsing-exposure with relevance retention",
             Hazard="Given exactly"),
        dict(Formulation="D", Inputs="Text, tier counts, hazard, relevance score",
             Objective="LambdaMART on hazard-discounted graded labels",
             Hazard="Given exactly"),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "table3_formulations.csv"), index=False)
    print("  wrote table3_formulations.csv")


# ============================================================================
def main():
    print("Schematics")
    fig1_framework()
    fig2_pipeline()
    fig3_protocol()
    print("Results figures")
    try:
        fig3_depth()
    except Exception as e:
        print("  [skip] fig3 needs scikit-learn and network access:", repr(e)[:80])
    fig4_battery()
    fig5_steelman()
    fig6_observability()
    fig7_conditionality()
    print("Tables")
    table3_formulations()
    try:
        table1_data()
    except Exception as e:
        print("  [skip] table1:", repr(e)[:80])
    print(f"\nAll output in {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
