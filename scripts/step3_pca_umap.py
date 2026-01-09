import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# -----------------------------
# Paths
# -----------------------------
EMB = "/fast/home/a/abejoy/projects/depmap_bulkrnabert/outputs/embeddings.npy"
TPM = "/fast/home/a/abejoy/data/depmap/processed/depmap_ensembl_aligned.npy"
SAMPLE_IDS = "/fast/home/a/abejoy/data/depmap/processed/sample_ids.txt"
META = "/fast/home/a/abejoy/data/depmap/metadata/Model.csv"

OUTDIR = "/fast/home/a/abejoy/projects/depmap_bulkrnabert/outputs/step3"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
E = np.load(EMB)
X = np.load(TPM)
ids = pd.read_csv(SAMPLE_IDS, header=None, names=["ModelID"])
meta = pd.read_csv(META)

df = ids.merge(meta, on="ModelID", how="left")
print("Matched metadata:", df.shape)

# -----------------------------
# PCA on embeddings
# -----------------------------
print("PCA on BulkRNABert embeddings")
E_scaled = StandardScaler().fit_transform(E)
pca_e = PCA(n_components=2)
E_pca = pca_e.fit_transform(E_scaled)

# -----------------------------
# PCA on raw TPM
# -----------------------------
print("PCA on raw TPM")
X_scaled = StandardScaler().fit_transform(X)
pca_x = PCA(n_components=2)
X_pca = pca_x.fit_transform(X_scaled)

# -----------------------------
# UMAP on embeddings
# -----------------------------
print("UMAP on embeddings")
umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
E_umap = umap_model.fit_transform(E)

# -----------------------------
# Plot helper
# -----------------------------
def plot_scatter(coords, color, title, fname):
    plt.figure(figsize=(6,5))
    s = plt.scatter(
        coords[:,0], coords[:,1],
        c=pd.factorize(color)[0],
        s=6, alpha=0.8
    )
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

# -----------------------------
# Plots
# -----------------------------
for field in ["OncotreeLineage", "OncotreePrimaryDisease", "TissueOrigin"]:
    plot_scatter(
        E_pca,
        df[field],
        f"BulkRNABert PCA colored by {field}",
        f"{OUTDIR}/bulkrna_pca_{field}.png"
    )

    plot_scatter(
        X_pca,
        df[field],
        f"Raw TPM PCA colored by {field}",
        f"{OUTDIR}/tpm_pca_{field}.png"
    )

    plot_scatter(
        E_umap,
        df[field],
        f"BulkRNABert UMAP colored by {field}",
        f"{OUTDIR}/bulkrna_umap_{field}.png"
    )

print("DONE — Step 3 complete")
