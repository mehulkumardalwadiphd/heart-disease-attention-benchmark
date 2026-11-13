"""
make_figures.py - draws every figure of the reworked Paper 01 from results/.
No training happens here; run code/pipeline.py first.

Colour = model family (validated palette): feature-token models blue,
single-token models orange, MLP aqua, classical models neutral grey.
Attention arms are filled markers, affine controls hollow, so the attention /
control distinction never rests on colour alone. Every model is also named on
its axis.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.patches import FancyBboxPatch       # noqa: E402
from sklearn.metrics import roc_curve               # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["LR", "DT", "RF", "XGB", "MLP", "ST-Attn", "ST-Affine",
          "FT-Attn", "FT-Affine"]
LABEL = {"LR": "Logistic regression", "DT": "Decision tree",
         "RF": "Random forest", "XGB": "XGBoost", "MLP": "MLP",
         "ST-Attn": "Single-token attention (T = 1)",
         "ST-Affine": "Single-token affine control",
         "FT-Attn": "Feature-token attention",
         "FT-Affine": "Feature-token affine control"}
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#5f5e5a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"LR": GREY, "DT": GREY, "RF": GREY, "XGB": GREY, "MLP": AQUA,
         "ST-Attn": ORANGE, "ST-Affine": ORANGE,
         "FT-Attn": BLUE, "FT-Affine": BLUE}
HOLLOW = {"ST-Affine", "FT-Affine"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
CAPTIONS = {}


def save(fig, name, caption):
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    CAPTIONS[name] = caption
    print("wrote", name)


def marker_kw(m, size=6):
    c = COLOR[m]
    return dict(marker="o", ms=size, mec=c, mew=1.6,
                mfc="white" if m in HOLLOW else c, color=c)


# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    R = "→"

    def box(x, y, w, h, text, fc="#f3f2ee", ec=INK2, size=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                color=INK, linespacing=1.3)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))

    ax.text(0.05, 4.95, "(a) Feature-token models (13 tokens)", fontsize=9,
            weight="bold", color=INK)
    for i, f in enumerate(["age", "sex", "cp", "...", "ca", "thal"]):
        box(0.05 + i * 0.52, 3.85, 0.47, 0.42, f, size=6.5)
    ax.text(1.6, 3.55, "13 clinical features", ha="center", fontsize=6.5, color=INK2)
    y = 4.06
    arrow(3.18, 3.5, y)
    box(3.55, 3.6, 2.0, 0.92,
        "Tokenizer\nnumeric: $x_j w_j + b_j$\ncategorical: $E_j[x_j] + b_j$", size=6.5)
    arrow(5.55, 5.85, y)
    box(5.9, 3.5, 3.1, 1.12, "", fc="#e8f0fb", ec=BLUE)
    ax.text(7.45, 4.4, "2 blocks", ha="center", fontsize=6.5, color=INK2)
    ax.text(7.45, 3.92, f"Mixer {R} + residual {R} LayerNorm\n"
            f"FFN {R} + residual {R} LayerNorm",
            ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.4)
    arrow(9.0, 9.3, y)
    box(9.35, 3.6, 2.55, 0.92,
        f"Mean over tokens\n{R} LayerNorm {R} ReLU\n{R} linear {R} p", size=6.5)
    box(5.9, 2.6, 1.5, 0.65, "Attention\n4 heads, 13 x 13", fc="white", ec=BLUE, size=6.3)
    box(7.5, 2.6, 1.5, 0.65, "Token-wise affine\n(control)", fc="white", ec=BLUE, size=6.3)
    ax.text(7.45, 2.33, "Mixer = one of the two; everything else identical",
            ha="center", fontsize=6.5, color=INK2)

    ax.text(0.05, 1.85, "(b) Single-token models (T = 1), the original design",
            fontsize=9, weight="bold", color=INK)
    y = 1.0
    box(0.05, 0.62, 1.8, 0.76, "Patient vector\n(22 inputs)")
    arrow(1.85, 2.15, y)
    box(2.2, 0.62, 2.5, 0.76, f"Dense 128 {R} Dense 64\n(ReLU, dropout 0.3)", size=6.5)
    arrow(4.7, 5.0, y)
    box(5.05, 0.5, 3.4, 1.0, "", fc="#fdeee7", ec=ORANGE)
    ax.text(6.75, 1.27, "one 64-d token", ha="center", fontsize=6.5, color=INK2)
    ax.text(6.75, 0.88, f"Mixer {R} + residual {R} LayerNorm", ha="center",
            fontsize=6.5, color=INK)
    arrow(8.45, 8.75, y)
    box(8.8, 0.62, 1.9, 0.76, f"Dense 32 {R} p")
    ax.text(6.75, 0.18, "Mixer = attention (softmax over a single key is 1)  or  a "
            "64 x 64 affine layer", ha="center", fontsize=6.5, color=INK2)
    save(fig, "fig1_architecture.png",
         "Model arms. (a) Feature-token models: each feature is one token; the mixer "
         "is multi-head attention or a token-wise affine control. (b) Single-token "
         "models from the original design (T = 1), with attention or an affine control.")


# ---------------------------------------------------------------------------
def fig_cv_auc(tag="onehot"):
    folds = pd.read_csv(RES / f"cv_folds_{tag}.csv")
    summ = pd.read_csv(RES / f"cv_summary_{tag}.csv").set_index("key")
    order = summ.loc[MODELS].sort_values("roc_auc_mean").index.tolist()
    # Text sizes raised for the second NJES round (Reviewer 2: the text of
    # this figure was too small once printed at full page width).
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rng = np.random.RandomState(0)
    for i, m in enumerate(order):
        v = folds.loc[folds.model == m, "roc_auc"].values
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=12, color=COLOR[m],
                   alpha=0.4, lw=0)
        s = summ.loc[m]
        ax.plot([s.roc_auc_ci_low, s.roc_auc_ci_high], [i, i], color=INK, lw=1.8)
        ax.plot(s.roc_auc_mean, i, **marker_kw(m, 8), zorder=5)
        ax.text(1.005, i, f"{s.roc_auc_mean:.3f} ± {s.roc_auc_sd:.3f}", va="center",
                fontsize=11, color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order], fontsize=11.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("ROC-AUC on the held-out fold (50 folds = 10 repeats x 5)",
                  fontsize=11.5)
    ax.grid(axis="y", visible=False)
    save(fig, f"fig2_cv_auc_{tag}.png",
         "ROC-AUC across the 50 held-out folds. Dots: folds; marker: mean; bar: "
         "corrected 95% CI; right: mean ± SD.")


def fig_pairwise(tag="onehot"):
    pw = pd.read_csv(RES / f"pairwise_tests_{tag}.csv")
    pw = pw[pw.metric == "roc_auc"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    n = len(pw)
    for i, r in pw.iterrows():
        y = n - 1 - i
        ax.plot([r.ci_low, r.ci_high], [y, y], color=INK, lw=1.4)
        ax.plot(r.mean_diff, y, **marker_kw(r.key_a, 6), zorder=5)
        ptxt = "p < 0.001" if r.p_holm < 0.001 else f"p = {r.p_holm:.3f}"
        ax.text(1.005, y, ptxt, va="center", fontsize=7.2, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(n), [f"{LABEL[a]} - {LABEL[b]}" for a, b in
                             zip(pw.key_a[::-1], pw.key_b[::-1])], fontsize=7.2)
    ax.set_xlabel("Difference in ROC-AUC (corrected 95% CI); right: Holm-adjusted p")
    ax.grid(axis="y", visible=False)
    save(fig, f"fig3_pairwise_{tag}.png",
         "Paired ROC-AUC differences over the same 50 folds, with the Nadeau-Bengio "
         "corrected 95% CI and Holm-adjusted p-values.")


def fig_roc(tag="onehot"):
    oof = pd.read_csv(RES / f"oof_predictions_{tag}.csv")
    grid = np.linspace(0, 1, 101)
    show = ["LR", "RF", "MLP", "ST-Attn", "FT-Attn", "FT-Affine"]
    styles = {"LR": "-", "RF": "--", "MLP": "-", "ST-Attn": "-", "FT-Attn": "-",
              "FT-Affine": "--"}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m in show:
        tprs = []
        for (_, _), g in oof[oof.model == m].groupby(["repeat", "fold"]):
            fpr, tpr, _ = roc_curve(g.y, g.prob)
            tprs.append(np.interp(grid, fpr, tpr))
        mean = np.mean(tprs, 0)
        mean[0] = 0
        ax.plot(grid, mean, styles[m], color=COLOR[m], lw=2 if m == "FT-Attn" else 1.4,
                label=LABEL[m])
    ax.plot([0, 1], [0, 1], ":", color=INK2, lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (mean over 50 folds)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_aspect("equal")
    save(fig, f"fig4_roc_{tag}.png", "Fold-averaged ROC curves for six models.")


def fig_learning(tag="onehot"):
    h = pd.read_csv(RES / f"training_histories_{tag}.csv")
    folds = pd.read_csv(RES / f"cv_folds_{tag}.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for row, m in enumerate(["FT-Attn", "ST-Attn"]):
        sub = h[(h.model == m) & (h.repeat == 0)]
        for col, (metric, lab) in enumerate([("loss", "BCE loss"), ("acc", "Accuracy")]):
            ax = axes[row, col]
            for f, g in sub.groupby("fold"):
                ax.plot(g.epoch, g[f"train_{metric}"], color=COLOR[m], lw=0.9, alpha=0.8)
                ax.plot(g.epoch, g[f"val_{metric}"], color=INK2, lw=0.9, alpha=0.8,
                        ls="--")
                be = folds[(folds.model == m) & (folds.repeat == 0) &
                           (folds.fold == f)].best_epoch.iloc[0]
                yv = g.loc[g.epoch == be, f"val_{metric}"]
                if len(yv):
                    ax.plot(be, yv.iloc[0], "o", ms=4, color=INK, zorder=5)
            ax.set_ylabel(lab)
            ax.set_title(f"{LABEL[m]}", fontsize=8.5, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles = [plt.Line2D([], [], color=INK2, lw=1.2,
                          label="fitting subset (solid, model colour)"),
               plt.Line2D([], [], color=INK2, lw=1.2, ls="--",
                          label="inner validation subset"),
               plt.Line2D([], [], marker="o", ls="", ms=4, color=INK,
                          label="restored epoch")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.2,
               bbox_to_anchor=(0.5, -0.04))
    save(fig, f"fig5_learning_{tag}.png",
         "Training histories for the five folds of repeat 1. Solid: fitting subset; "
         "dashed: inner validation subset; dot: epoch whose weights were restored.")


# ---------------------------------------------------------------------------
def fig_shap():
    imp = pd.read_csv(RES / "shap_importance.csv", index_col="model")
    names = {v: k for k, v in LABEL.items()}
    full = {"Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF"}
    keys = [full.get(i, names.get(i, i)) for i in imp.index]
    imp.index = [LABEL.get(k, k) for k in keys]
    share = imp.div(imp.sum(axis=1), axis=0)
    ranks = imp.rank(axis=1, ascending=False, method="min").astype(int)
    order = share.mean().sort_values(ascending=False).index
    share, ranks = share[order], ranks[order]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    im = ax.imshow(share.values, cmap="Blues", aspect="auto", vmin=0)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, ranks.values[i, j], ha="center", va="center", fontsize=7,
                    color="white" if v > share.values.max() * 0.55 else INK)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(share)), share.index)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Share of total mean |SHAP|")
    save(fig, "fig6_shap_importance.png",
         "Kernel SHAP importance per model, same explainer, background and patients "
         "(every patient explained out of fold, 3 repeats). Shade: share of the model's "
         "total mean |SHAP|; number: rank.")


def fig_rank_agreement():
    ag = pd.read_csv(RES / "shap_rank_agreement.csv")
    keys = [m for m in MODELS if m in set(ag.key_a) | set(ag.key_b)]
    n = len(keys)
    M = np.eye(n)
    lo = np.eye(n)
    for _, r in ag.iterrows():
        i, j = keys.index(r.key_a), keys.index(r.key_b)
        M[i, j] = M[j, i] = r.rho
        lo[i, j] = lo[j, i] = r.ci_low
    mask = np.triu(np.ones_like(M, bool))
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(np.where(mask, np.nan, M), cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(i):
            ax.text(j, i, f"{M[i, j]:.2f}\n[{lo[i, j]:.2f}]", ha="center", va="center",
                    fontsize=6.3, color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(n), [LABEL[k] for k in keys], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(n), [LABEL[k] for k in keys], fontsize=7.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman ρ between importance rankings")
    save(fig, "fig7_rank_agreement.png",
         "Agreement of SHAP importance rankings between models (Spearman ρ; "
         "bracket: lower bound of the patient-bootstrap 95% CI).")


def fig_attention():
    tab = pd.read_csv(RES / "attention_vs_shap.csv")
    summ = pd.read_csv(RES / "attention_vs_shap_summary.csv").iloc[0]
    maps = pd.read_csv(RES / "attention_maps_onehot.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                               gridspec_kw={"width_ratios": [1, 1.1]})
    a.scatter(tab.mean_attention_received, tab.mean_abs_shap, s=28, color=BLUE,
              edgecolor="white", lw=1.5, zorder=3)
    # hand-placed label offsets (points) where the cloud is crowded
    offs = {"thalach": (-34, -9), "oldpeak": (-36, 4), "sex": (5, 3),
            "chol": (5, 2), "restecg": (-38, -3), "age": (5, -8),
            "trestbps": (-42, 2), "exang": (5, 0)}
    for _, r in tab.iterrows():
        a.annotate(r.feature, (r.mean_attention_received, r.mean_abs_shap),
                   xytext=offs.get(r.feature, (4, 2)), textcoords="offset points",
                   fontsize=7, color=INK)
    a.axvline(summ.uniform_attention, color=INK2, lw=0.8, ls=":")
    a.text(summ.uniform_attention, a.get_ylim()[1], " uniform", fontsize=6.5,
           color=INK2, va="top")
    a.set_xlabel("Mean attention received")
    a.set_ylabel("Mean |SHAP| (same model, same patients)")
    a.set_title(f"(a) ρ = {summ.rho_global:.2f} [{summ.ci_low:.2f}, {summ.ci_high:.2f}]",
                fontsize=8.5, loc="left", color=INK)
    toks = [c for c in maps.columns if c not in ("layer", "query")]
    mean = maps.groupby("query")[toks].mean().loc[toks].values
    dev = mean - 1 / len(toks)                       # deviation from uniform
    lim = np.abs(dev).max()
    im = b.imshow(dev, cmap="RdBu_r", vmin=-lim, vmax=lim)
    b.set_xticks(range(len(toks)), toks, rotation=90, fontsize=7)
    b.set_yticks(range(len(toks)), toks, fontsize=7)
    b.set_xlabel("Key (attended-to feature)")
    b.set_ylabel("Query")
    b.grid(False)
    b.set_title("(b) Attention minus uniform (1/13), both layers", fontsize=8.5,
                loc="left", color=INK)
    fig.colorbar(im, ax=b, fraction=0.045, pad=0.02).set_label(
        "Mean attention − 1/13", fontsize=7.5)
    save(fig, "fig8_attention.png",
         "Feature-token attention. (a) Attention received vs mean |SHAP| per feature; "
         "dotted: uniform attention (1/13). (b) Mean attention map over 50 folds, "
         "shown as the deviation from uniform attention.")


# ---------------------------------------------------------------------------
def fig_external():
    ext = pd.read_csv(RES / "external_summary.csv")
    ext = ext[ext.feature_set == "ext"]
    cv = pd.read_csv(RES / "cv_summary_ext.csv").set_index("key")
    sites = ["Hungarian", "Switzerland", "VA Long Beach"]
    order = cv.loc[MODELS].sort_values("roc_auc_mean").index.tolist()
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 3.6), sharey=True)
    panels = [("Cleveland (internal CV)", None)] + [(s, s) for s in sites]
    for ax, (title, site) in zip(axes, panels):
        for i, m in enumerate(order):
            if site is None:
                r = cv.loc[m]
                lo, hi, mu = r.roc_auc_ci_low, r.roc_auc_ci_high, r.roc_auc_mean
            else:
                r = ext[(ext.site == site) & (ext.key == m)].iloc[0]
                lo, hi, mu = r.roc_auc_ci_low, r.roc_auc_ci_high, r.roc_auc_mean
            ax.plot([lo, hi], [i, i], color=INK, lw=1.3)
            ax.plot(mu, i, **marker_kw(m, 5.5), zorder=5)
        sub = ext[ext.site == site] if site else None
        if site:
            n, pos = int(sub.n.iloc[0]), int(sub.positive.iloc[0])
            title = f"{title}\nn = {n}, {n - pos} without CAD"
        ax.set_title(title, fontsize=8, color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.45, 1.0)
    axes[0].set_yticks(range(len(order)), [LABEL[m] for m in order], fontsize=7.5)
    fig.supxlabel("ROC-AUC, 8 shared features (95% CI)", fontsize=8.5)
    save(fig, "fig9_external.png",
         "Trained on Cleveland, tested on three other UCI sites, on the 8 features "
         "recorded at every site. External: mean over 10 seeds, patient-bootstrap "
         "95% CI; Cleveland: 10 x 5 CV on the same 8 features.")


def fig_sensitivity():
    a = pd.read_csv(RES / "cv_summary_onehot.csv").set_index("key")
    b = pd.read_csv(RES / "cv_summary_integer.csv").set_index("key")
    order = a.loc[MODELS].sort_values("roc_auc_mean").index.tolist()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, m in enumerate(order):
        x0, x1 = a.loc[m, "roc_auc_mean"], b.loc[m, "roc_auc_mean"]
        ax.plot([x0, x1], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.plot(x0, i, "o", ms=7, color=COLOR[m], zorder=3)
        ax.plot(x1, i, "D", ms=6, mfc="white", mec=COLOR[m], mew=1.5, zorder=3)
    ax.plot([], [], "o", color=INK2, label="one-hot nominal (main)")
    ax.plot([], [], "D", mfc="white", mec=INK2, label="z-scored integer codes")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set_xlabel("Mean ROC-AUC over 50 folds")
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_encoding_sensitivity.png",
         "Encoding sensitivity: one-hot nominal variables (main) vs the original "
         "z-scored integer codes, same 50 folds.")


def fig_ablation():
    ab = pd.read_csv(RES / "ablation_summary.csv")
    piv = ab.pivot(index="heads", columns="attn_dropout", values="roc_auc_mean")
    sd = ab.pivot(index="heads", columns="attn_dropout", values="roc_auc_sd")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    # Fixed range (0.85-0.95, about one fold SD either side) so that a spread
    # of a few thousandths is not painted as a strong pattern.
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=0.85, vmax=0.95)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}\n±{sd.values[i, j]:.3f}", ha="center", va="center",
                    fontsize=7.5, color="white" if v > 0.92 else INK)
    ax.set_xticks(range(piv.shape[1]), [f"{c:.1f}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("Attention dropout")
    ax.set_ylabel("Heads")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("Mean ROC-AUC")
    save(fig, "fig11_ablation.png",
         "Head count x attention dropout for the feature-token attention model, "
         "same 50 folds (mean ± SD ROC-AUC; colour scale fixed at 0.85-0.95).")


def main():
    fig_architecture()
    fig_cv_auc("onehot")
    fig_pairwise("onehot")
    fig_roc("onehot")
    fig_learning("onehot")
    fig_shap()
    fig_rank_agreement()
    fig_attention()
    fig_external()
    fig_sensitivity()
    fig_ablation()
    json.dump(CAPTIONS, open(FIG / "captions.json", "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
