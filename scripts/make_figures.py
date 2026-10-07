#!/usr/bin/env python3
"""
Figures and per-lineage results for the BulkRNABert vs expression-PCA comparison.

Uses the same evaluation as step3_4_downstream_depmap.py (same labels, filtering,
kNN settings and random seeds), so the summary metrics are identical. It adds:

  figures/metrics_comparison.png   summary metrics for the three representations
  figures/umap_by_lineage.png      UMAP of embeddings vs expression, one panel per lineage
  figures/per_lineage_recall.png   recall per lineage, embeddings vs expression
  results/metrics_summary.csv
  results/per_lineage_recall.csv

Usage:
  python scripts/make_figures.py                      # looks for inputs under --base
  python scripts/make_figures.py --base /path/to/data # folder that contains all inputs
  python scripts/make_figures.py --emb embeddings.npy --expr depmap_ensembl_aligned.npy \\
      --raw-expr OmicsExpressionProteinCodingGenesTPMLogp1.csv --meta Model.csv
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import umap

# Evaluation settings (identical to step3_4_downstream_depmap.py)
LABEL_COL = "OncotreeLineage"
MIN_CLASS_N = 10
KNN_K = 10
CV_SPLITS = 5
HVG_N_GENES = 2000
N_PCS = 20
SEED = 0

# One fixed colour per representation, used in every figure
C_EMB, C_ALL, C_HVG = "#2a78d6", "#eb6834", "#1baf7a"
C_BG, C_GRID, C_TEXT, C_MUTED = "#d5d4ce", "#e6e5e0", "#0b0b0b", "#52514e"

NAME_EMB = "BulkRNABert embeddings"
NAME_ALL = "Expression PCA, all genes"
NAME_HVG = "Expression PCA, 2,000 variable genes"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT,
    "xtick.color": C_MUTED, "ytick.color": C_TEXT, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def find_input(explicit, filename, base):
    """Return the explicit path if given, otherwise search for filename under base."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise SystemExit(f"ERROR: file not found: {p}")
        return p
    hits = sorted(Path(base).expanduser().rglob(filename))
    if not hits:
        raise SystemExit(f"ERROR: could not find '{filename}' under {base}. Pass its path explicitly (see --help).")
    if len(hits) > 1:
        print(f"NOTE: several '{filename}' found, using {hits[0]}")
    return hits[0]


def knn_purity(X, y):
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="cosine").fit(X)
    neigh = nn.kneighbors(return_distance=False)[:, 1:]
    return float(np.mean(y[neigh] == y[:, None]))


def knn_cv_predict(X, y):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    clf = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine", weights="distance")
    return cross_val_predict(clf, X, y, cv=cv)


def plot_metrics(metrics, out_png):
    cols = [("knn_purity", "kNN purity"), ("knn_accuracy", "Accuracy"),
            ("knn_balanced_accuracy", "Balanced accuracy"), ("knn_macroF1", "Macro F1")]
    order = [NAME_HVG, NAME_ALL, NAME_EMB]          # drawn bottom to top
    colors = {NAME_EMB: C_EMB, NAME_ALL: C_ALL, NAME_HVG: C_HVG}
    m = metrics.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.5), sharey=True)
    for ax, (col, title) in zip(axes, cols):
        vals = m[col].values
        ax.barh(order, vals, height=0.55, color=[colors[o] for o in order])
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, f"{v:.2f}", va="center", color=C_TEXT)
        ax.set_title(title, loc="left", fontsize=10.5, color=C_TEXT)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    fig.suptitle("Lineage classification of DepMap cell lines (kNN, 5-fold cross-validation)",
                 x=0.01, ha="left", fontsize=11.5, color=C_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_umap_facets(U_emb, U_hvg, y, out_png, n_show=6):
    top = pd.Series(y).value_counts().index[:n_show]
    rows = [("BulkRNABert embeddings", U_emb, C_EMB), ("Expression PCA\n(2,000 variable genes)", U_hvg, C_HVG)]
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5.6))
    for r, (row_name, U, color) in enumerate(rows):
        for c, lineage in enumerate(top):
            ax = axes[r, c]
            sel = y == lineage
            ax.scatter(U[~sel, 0], U[~sel, 1], s=3, color=C_BG, linewidths=0)
            ax.scatter(U[sel, 0], U[sel, 1], s=9, color=color, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(C_GRID)
            if r == 0:
                ax.set_title(f"{lineage}\n(n = {int(sel.sum())})", fontsize=9.5, color=C_TEXT)
            if c == 0:
                ax.set_ylabel(row_name, fontsize=9.5, color=C_TEXT)
    fig.suptitle(f"UMAP of {len(y):,} cell lines; each panel highlights one of the {n_show} largest lineages",
                 x=0.01, ha="left", fontsize=11.5, color=C_TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_per_lineage(per, out_png):
    per = per.sort_values("recall_expression_hvg")
    ypos = np.arange(len(per))
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(per) + 1.6))
    ax.hlines(ypos, per["recall_embeddings"], per["recall_expression_hvg"], color=C_BG, linewidth=2, zorder=1)
    ax.scatter(per["recall_embeddings"], ypos, s=46, color=C_EMB, zorder=2, label=NAME_EMB,
               edgecolors="white", linewidths=1)
    ax.scatter(per["recall_expression_hvg"], ypos, s=46, color=C_HVG, zorder=2, label=NAME_HVG,
               edgecolors="white", linewidths=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{l} ({n})" for l, n in zip(per["lineage"], per["n_cell_lines"])])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Recall (share of a lineage's cell lines classified correctly)")
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handletextpad=0.3)
    ax.set_title("Recall per lineage (number of cell lines in brackets)", loc="left",
                 fontsize=11.5, color=C_TEXT, pad=28)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--base", default=str(repo.parent), help="folder searched for the input files (default: parent of the repository)")
    ap.add_argument("--emb", help="embeddings.npy")
    ap.add_argument("--expr", help="depmap_ensembl_aligned.npy")
    ap.add_argument("--raw-expr", help="OmicsExpressionProteinCodingGenesTPMLogp1.csv (for the sample order)")
    ap.add_argument("--meta", help="DepMap Model.csv")
    ap.add_argument("--outdir", default=str(repo), help="where figures/ and results/ are written (default: repository)")
    a = ap.parse_args()

    emb_p = find_input(a.emb, "embeddings.npy", a.base)
    expr_p = find_input(a.expr, "depmap_ensembl_aligned.npy", a.base)
    raw_p = find_input(a.raw_expr, "OmicsExpressionProteinCodingGenesTPMLogp1.csv", a.base)
    meta_p = find_input(a.meta, "Model.csv", a.base)
    for name, p in [("embeddings", emb_p), ("expression", expr_p), ("raw expression", raw_p), ("metadata", meta_p)]:
        print(f"{name:>15}: {p}")

    fig_dir, res_dir = Path(a.outdir) / "figures", Path(a.outdir) / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    E = np.load(emb_p)
    X = np.load(expr_p, mmap_mode="r")
    ids = pd.read_csv(raw_p, usecols=[0]).iloc[:, 0].astype(str).tolist()
    assert len(ids) == E.shape[0] == X.shape[0], "row count mismatch between sample IDs, embeddings and expression"

    meta = pd.read_csv(meta_p, low_memory=False)
    meta = meta.set_index(meta["ModelID"].astype(str))
    labels = np.asarray(meta.reindex(ids)[LABEL_COL].fillna("Unknown"), dtype=str)

    counts = pd.Series(labels).value_counts()
    keep = np.isin(labels, counts[counts >= MIN_CLASS_N].index)
    y = labels[keep]
    lineages = np.array(sorted(set(y)))
    print(f"{keep.sum()} cell lines, {len(lineages)} lineages with at least {MIN_CLASS_N} cell lines")

    E_f = E[keep]
    X_f = np.asarray(X[keep], dtype=np.float32)
    X_all = PCA(n_components=N_PCS, random_state=SEED).fit_transform(X_f)
    top_genes = np.argsort(X_f.var(axis=0))[-HVG_N_GENES:]
    X_hvg = PCA(n_components=N_PCS, random_state=SEED).fit_transform(X_f[:, top_genes])

    reps = [(NAME_EMB, E_f), (NAME_ALL, X_all), (NAME_HVG, X_hvg)]
    rows, preds = [], {}
    for name, R in reps:
        pred = knn_cv_predict(R, y)
        preds[name] = pred
        rows.append({
            "method": name, "N": int(len(y)), "n_classes": int(len(lineages)),
            "knn_purity": knn_purity(R, y),
            "knn_accuracy": accuracy_score(y, pred),
            "knn_balanced_accuracy": balanced_accuracy_score(y, pred),
            "knn_macroF1": f1_score(y, pred, average="macro"),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(res_dir / "metrics_summary.csv", index=False)
    print("\n" + metrics.round(3).to_string(index=False))

    per = pd.DataFrame({
        "lineage": lineages,
        "n_cell_lines": [int((y == l).sum()) for l in lineages],
        "recall_embeddings": recall_score(y, preds[NAME_EMB], labels=lineages, average=None),
        "recall_expression_allgenes": recall_score(y, preds[NAME_ALL], labels=lineages, average=None),
        "recall_expression_hvg": recall_score(y, preds[NAME_HVG], labels=lineages, average=None),
    })
    per["difference_hvg_minus_embeddings"] = per["recall_expression_hvg"] - per["recall_embeddings"]
    per = per.sort_values("n_cell_lines", ascending=False)
    per.to_csv(res_dir / "per_lineage_recall.csv", index=False)
    print("\n" + per.round(2).to_string(index=False))

    print("\nRunning UMAP ...")
    kw = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
    U_emb = umap.UMAP(**kw).fit_transform(E_f)
    U_hvg = umap.UMAP(**kw).fit_transform(X_hvg)

    plot_metrics(metrics, fig_dir / "metrics_comparison.png")
    plot_umap_facets(U_emb, U_hvg, y, fig_dir / "umap_by_lineage.png")
    plot_per_lineage(per, fig_dir / "per_lineage_recall.png")
    print(f"\nDone. Figures in {fig_dir}, tables in {res_dir}")


if __name__ == "__main__":
    main()
