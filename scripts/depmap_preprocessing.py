#!/usr/bin/env python3
"""
DepMap → BulkRNABert preprocessing

Input:
  - DepMap OmicsExpressionProteinCodingGenesTPMLogp1.csv
    (rows=samples / DepMap model IDs, cols="GENE (ENTREZ_ID)")
  - NCBI gene2ensembl mapping (tab-delimited)

Output:
  - depmap_ensembl_aligned.npy   (shape: n_samples x n_genes, float32)
  - tpm_distribution.png
  - housekeeping_genes.png

Notes:
  - Maps Entrez → Ensembl (human only, tax_id=9606)
  - Aggregates duplicate Ensembl IDs by mean
  - Aligns gene order to BulkRNABert common_gene_id.txt
  - Zero-fills missing genes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paths (edit if needed)
# -------------------------
RAW_EXPR = "/fast/home/a/abejoy/data/depmap/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
GENE2ENS = "/fast/home/a/abejoy/data/gene_mapping/gene2ensembl"
GENE_LIST = "/fast/home/a/abejoy/data/bulkrnabert/multiomics-open-research/data/bulkrnabert/common_gene_id.txt"

OUT_DIR = Path("/fast/home/a/abejoy/data/depmap/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATRIX = OUT_DIR / "depmap_ensembl_aligned.npy"
OUT_TPM_PLOT = OUT_DIR / "tpm_distribution.png"
OUT_HK_PLOT = OUT_DIR / "housekeeping_genes.png"


# -------------------------
# Helpers
# -------------------------
def extract_entrez(col: str) -> str:
    """Extract Entrez ID from 'GENE (1234)' style strings; fallback to raw string."""
    if "(" in col and ")" in col:
        return col.split("(")[1].split(")")[0].strip()
    return col.strip()


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


# -------------------------
# Main
# -------------------------
print("Starting DepMap → BulkRNABert preprocessing")
print("RAW_EXPR:", RAW_EXPR)
print("GENE2ENS:", GENE2ENS)
print("GENE_LIST:", GENE_LIST)
print("OUT_DIR:", OUT_DIR)

# -------------------------
# Load DepMap expression
# -------------------------
if not Path(RAW_EXPR).exists():
    die(f"Missing RAW_EXPR: {RAW_EXPR}")

expr = pd.read_csv(RAW_EXPR)
if expr.shape[1] < 2:
    die("Expression file appears to have <2 columns. Is RAW_EXPR correct?")

expr = expr.rename(columns={expr.columns[0]: "sample_id"}).set_index("sample_id")

# Parse Entrez IDs from headers
expr.columns = [extract_entrez(c) for c in expr.columns]

# Validate columns are numeric Entrez IDs (best-effort)
bad = [c for c in expr.columns if not c.isdigit()]
print("Expression shape:", expr.shape)
print("Non-numeric gene columns:", len(bad))
if bad:
    print("Example bad columns:", bad[:5])

# -------------------------
# Load gene2ensembl mapping
# -------------------------
if not Path(GENE2ENS).exists():
    die(f"Missing GENE2ENS: {GENE2ENS}")

map_df = pd.read_csv(
    GENE2ENS,
    sep="\t",
    usecols=["#tax_id", "GeneID", "Ensembl_gene_identifier"],
    dtype={
        "#tax_id": "int32",
        "GeneID": "int64",
        "Ensembl_gene_identifier": "string",
    },
)

print("Mapping table loaded:", map_df.shape)

# Human only
map_df = map_df[map_df["#tax_id"] == 9606].rename(
    columns={
        "#tax_id": "tax_id",
        "GeneID": "entrez_id",
        "Ensembl_gene_identifier": "ensembl_id",
    }
)

map_df = map_df[["entrez_id", "ensembl_id"]].dropna()
print("Mapping rows (human):", map_df.shape)

# -------------------------
# Entrez → Ensembl mapping
# -------------------------
expr_t = expr.T.reset_index().rename(columns={"index": "entrez_id"})

# Coerce entrez_id to int64; drop non-numeric if present
expr_t["entrez_id_raw"] = expr_t["entrez_id"].astype(str)
expr_t = expr_t[expr_t["entrez_id_raw"].str.isdigit()].copy()
expr_t["entrez_id"] = expr_t["entrez_id_raw"].astype("int64")
expr_t = expr_t.drop(columns=["entrez_id_raw"])

merged = expr_t.merge(map_df, on="entrez_id", how="left")
unmapped = int(merged["ensembl_id"].isna().sum())
print("Unmapped genes (after filtering non-numeric headers):", unmapped)

# -------------------------
# Aggregate duplicate Ensembl IDs (mean)
# -------------------------
expr_ensembl = (
    merged.dropna(subset=["ensembl_id"])
    .drop(columns=["entrez_id"])
    .groupby("ensembl_id")
    .mean()
)

print("After aggregation (unique Ensembl IDs x samples):", expr_ensembl.shape)

# -------------------------
# Align to BulkRNABert gene list
# -------------------------
if not Path(GENE_LIST).exists():
    die(f"Missing GENE_LIST: {GENE_LIST}")

with open(GENE_LIST) as f:
    gene_list = [g.strip() for g in f if g.strip()]

if len(gene_list) == 0:
    die("GENE_LIST is empty.")

present = expr_ensembl.index.intersection(gene_list)
missing = set(gene_list) - set(expr_ensembl.index)
print(f"Genes in model list: {len(gene_list)}")
print(f"Present after mapping: {len(present)}")
print(f"Missing (zero-filled): {len(missing)}")
print(f"Coverage: {len(present)/len(gene_list):.4f}")

expr_aligned = expr_ensembl.reindex(gene_list, fill_value=0.0)

print("Final aligned shape (genes x samples):", expr_aligned.shape)
print("NaNs:", int(np.isnan(expr_aligned.values).sum()))

# Hard check: exact ordering
if list(expr_aligned.index) != gene_list:
    die("Gene order mismatch after reindex — this should not happen.")

# -------------------------
# Save matrix (samples x genes)
# -------------------------
X = expr_aligned.T.values.astype(np.float32)

print("Saved matrix shape (samples x genes):", X.shape)
print("X min/max/mean:", float(X.min()), float(X.max()), float(X.mean()))
print("Percent zeros:", float((X == 0).mean() * 100.0), "%")

np.save(OUT_MATRIX, X)
print("Saved to:", OUT_MATRIX)

# =========================================================
# Sanity-check plots
# =========================================================
print("Generating sanity-check plots...")

# TPM distribution (log1p)
plt.figure(figsize=(6, 4))
plt.hist(X.flatten(), bins=100)
plt.xlabel("log1p(TPM)")
plt.ylabel("Frequency")
plt.title("DepMap expression distribution (log1p TPM)")
plt.tight_layout()
plt.savefig(OUT_TPM_PLOT, dpi=150)
plt.close()
print("Saved TPM distribution plot:", OUT_TPM_PLOT)

# Housekeeping genes (if present)
# GAPDH: ENSG00000111640
# ACTB : ENSG00000075624
gene_to_idx = {g: i for i, g in enumerate(gene_list)}

plt.figure(figsize=(6, 4))
plotted_any = False

for gene, label in [("ENSG00000111640", "GAPDH"), ("ENSG00000075624", "ACTB")]:
    if gene not in gene_to_idx:
        print(f"WARNING: {label} ({gene}) not found in gene_list; skipping.")
        continue
    idx = gene_to_idx[gene]
    plt.hist(X[:, idx], bins=50, alpha=0.6, label=label)
    plotted_any = True

if plotted_any:
    plt.xlabel("log1p(TPM)")
    plt.ylabel("Frequency")
    plt.title("Housekeeping gene expression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HK_PLOT, dpi=150)
    print("Saved housekeeping gene plot:", OUT_HK_PLOT)
else:
    plt.close()
    print("WARNING: No housekeeping genes plotted; plot not saved.")

print("DONE — preprocessing + sanity checks complete")
