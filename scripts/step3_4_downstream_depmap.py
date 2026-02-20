#!/usr/bin/env python3
"""
DepMap BulkRNABert embeddings: minimal downstream eval (biologically defensible)

Core question:
  Do embeddings preserve biologically meaningful structure (OncotreeLineage)
  compared to standard expression baselines?

Outputs (outputs/analysis_minimal/):
  - umap_embeddings_lineage.png
  - pca_embeddings_lineage.png
  - pca_expr_allgenes_lineage.png
  - pca_expr_hvg_lineage.png
  - metrics_summary.csv

Method choices (why correct):
  - Use OncotreeLineage as primary label.
  - Filter rare lineages (< MIN_CLASS_N) before CV (avoids invalid folds).
  - Compare vs expression PCA(20) and HVG-PCA(20) (standard transcriptomics baselines).
  - Report accuracy + balanced accuracy + macro-F1 + kNN purity (robust + interpretable).
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

try:
    import umap  # umap-learn
except Exception as e:
    raise RuntimeError("umap-learn not installed. Install: pip install umap-learn") from e


# -----------------------
# Paths (edit if needed)
# -----------------------
PROJ = "/fast/home/a/abejoy/projects/depmap_bulkrnabert"
RAW_EXPR = "/fast/home/a/abejoy/data/depmap/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
META = "/fast/home/a/abejoy/data/depmap/metadata/Model.csv"

EMB_NPY = f"{PROJ}/outputs/embeddings.npy"
X_NPY = "/fast/home/a/abejoy/data/depmap/processed/depmap_ensembl_aligned.npy"  # (N,19062), log2(TPM+1)

OUT_DIR = Path(f"{PROJ}/outputs/analysis_minimal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Evaluation parameters
# -----------------------
LABEL_COL = "OncotreeLineage"
MIN_CLASS_N = 10          # filter rare lineages for valid CV
KNN_K = 10                # kNN for purity + classifier
CV_SPLITS = 5             # after filtering >=10, 5-fold is safe
HVG_N_GENES = 2000        # standard baseline


# -----------------------
# Helpers
# -----------------------
def save_scatter(coords, labels, title, out_png, point_size=7):
    labels = pd.Series(labels).fillna("Unknown").astype("category")
    codes = labels.cat.codes
    plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=codes, s=point_size, cmap="tab20", alpha=0.85)
    plt.title(title)
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def knn_purity(X, y_str, k=10, metric="cosine") -> float:
    y = np.asarray(y_str, dtype=str)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    neigh = nn.kneighbors(return_distance=False)[:, 1:]  # drop self
    return float(np.mean([np.mean(y[neigh[i]] == y[i]) for i in range(neigh.shape[0])]))


def knn_cv_metrics(X, y_codes, k=10, cv_splits=5, metric="cosine"):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric, weights="distance")
    pred = cross_val_predict(clf, X, y_codes, cv=cv)

    acc = accuracy_score(y_codes, pred)
    bal = balanced_accuracy_score(y_codes, pred)
    f1m = f1_score(y_codes, pred, average="macro")
    return float(acc), float(bal), float(f1m)


def main():
    # Load embeddings + expression
    E = np.load(EMB_NPY)  # (N,256)
    X = np.load(X_NPY, mmap_mode="r")  # (N,19062)
    print("Embeddings:", E.shape, E.dtype)
    print("Expression:", X.shape, X.dtype)

    # Reconstruct sample order (ModelID / ACH- IDs)
    sample_ids = pd.read_csv(RAW_EXPR, usecols=[0]).iloc[:, 0].astype(str).tolist()
    assert len(sample_ids) == E.shape[0] == X.shape[0], "Row count mismatch between ids/embeddings/expression"

    # Load metadata and align
    meta = pd.read_csv(META, low_memory=False)
    meta = meta.set_index(meta["ModelID"].astype(str))
    meta_aligned = meta.reindex(sample_ids).reset_index(drop=True)

    # Labels
    y_all = pd.Series(meta_aligned.get(LABEL_COL, "Unknown")).fillna("Unknown").astype("category")
    counts = y_all.value_counts()

    # Filter rare classes for valid CV
    keep = y_all.isin(counts[counts >= MIN_CLASS_N].index).values
    y = y_all[keep].astype("category")
    y_codes = y.cat.codes.values
    print(f"Filtered {LABEL_COL}: >= {MIN_CLASS_N} samples -> {keep.sum()} samples, {y.nunique()} classes")

    # Subset data
    E_f = E[keep]
    X_f_full = np.asarray(X[keep], dtype=np.float32)  # expression subset in RAM (1657x19062 is fine)

    # Baseline 1: PCA(20) on all genes
    X_pca20 = PCA(n_components=20, random_state=0).fit_transform(X_f_full)

    # Baseline 2: HVG-PCA(20)
    gene_var = X_f_full.var(axis=0)
    top = np.argsort(gene_var)[-HVG_N_GENES:]
    X_hvg = X_f_full[:, top]
    X_hvg_pca20 = PCA(n_components=20, random_state=0).fit_transform(X_hvg)

    # PCA plot (embeddings + baselines)
    E_pca2 = PCA(n_components=2, random_state=0).fit_transform(E_f)
    X_pca2 = X_pca20[:, :2]
    X_hvg_pca2 = X_hvg_pca20[:, :2]

    save_scatter(E_pca2, y, f"PCA(2) embeddings — {LABEL_COL}", OUT_DIR / "pca_embeddings_lineage.png", point_size=6)
    save_scatter(X_pca2, y, f"PCA(2) expr all-genes — {LABEL_COL}", OUT_DIR / "pca_expr_allgenes_lineage.png", point_size=6)
    save_scatter(X_hvg_pca2, y, f"PCA(2) expr HVG({HVG_N_GENES}) — {LABEL_COL}", OUT_DIR / "pca_expr_hvg_lineage.png", point_size=6)

    # UMAP on embeddings (main visualization)
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(E_f)
    save_scatter(U, y, f"UMAP embeddings — {LABEL_COL}", OUT_DIR / "umap_embeddings_lineage.png", point_size=7)

    # Quantitative metrics (robust + interpretable)
    purity_E = knn_purity(E_f, y.astype(str).values, k=KNN_K, metric="cosine")
    purity_X = knn_purity(X_pca20, y.astype(str).values, k=KNN_K, metric="cosine")
    purity_H = knn_purity(X_hvg_pca20, y.astype(str).values, k=KNN_K, metric="cosine")

    acc_E, bal_E, f1_E = knn_cv_metrics(E_f, y_codes, k=KNN_K, cv_splits=CV_SPLITS, metric="cosine")
    acc_X, bal_X, f1_X = knn_cv_metrics(X_pca20, y_codes, k=KNN_K, cv_splits=CV_SPLITS, metric="cosine")
    acc_H, bal_H, f1_H = knn_cv_metrics(X_hvg_pca20, y_codes, k=KNN_K, cv_splits=CV_SPLITS, metric="cosine")

    metrics = pd.DataFrame([
        {
            "label": LABEL_COL,
            "MIN_CLASS_N": MIN_CLASS_N,
            "N": int(E_f.shape[0]),
            "n_classes": int(y.nunique()),
            "kNN_k": KNN_K,
            "method": "BulkRNABert_embeddings",
            "knn_purity": purity_E,
            "knn_accuracy": acc_E,
            "knn_balanced_accuracy": bal_E,
            "knn_macroF1": f1_E,
        },
        {
            "label": LABEL_COL,
            "MIN_CLASS_N": MIN_CLASS_N,
            "N": int(E_f.shape[0]),
            "n_classes": int(y.nunique()),
            "kNN_k": KNN_K,
            "method": "Expr_PCA20_allgenes",
            "knn_purity": purity_X,
            "knn_accuracy": acc_X,
            "knn_balanced_accuracy": bal_X,
            "knn_macroF1": f1_X,
        },
        {
            "label": LABEL_COL,
            "MIN_CLASS_N": MIN_CLASS_N,
            "N": int(E_f.shape[0]),
            "n_classes": int(y.nunique()),
            "kNN_k": KNN_K,
            "method": f"Expr_HVG{HVG_N_GENES}_PCA20",
            "knn_purity": purity_H,
            "knn_accuracy": acc_H,
            "knn_balanced_accuracy": bal_H,
            "knn_macroF1": f1_H,
        },
    ])
    metrics.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    print("\n=== DONE ===")
    print("Outputs:", OUT_DIR)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
