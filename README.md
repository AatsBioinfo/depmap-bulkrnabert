depmap-bulkrnabert

This repository provides a reproducible pipeline to extract and evaluate **BulkRNABert embeddings** from DepMap bulk RNA-seq data.

Model source:
[https://github.com/instadeepai/multiomics-open-research](https://github.com/instadeepai/multiomics-open-research)

---

# 🎯 Objective

To evaluate whether pretrained gene-expression embeddings (BulkRNABert) preserve meaningful biological structure in DepMap cancer cell lines.

Specifically:

* Do embeddings reflect tissue lineage?
* How do embeddings compare to standard gene expression baselines (PCA, HVG)?

---

# 🧬 Pipeline Overview

The workflow consists of three main stages:

1. Preprocessing DepMap expression data
2. Extracting BulkRNABert embeddings
3. Downstream biological evaluation

---

# 1️⃣ Preprocessing

Script: `scripts/preprocess_depmap.py`

### Input

* `OmicsExpressionProteinCodingGenesTPMLogp1.csv` (DepMap)
* NCBI `gene2ensembl` mapping file
* `common_gene_id.txt` from BulkRNABert

### Processing steps

* Extract Entrez IDs from gene headers
* Map Entrez → Ensembl (human only)
* Aggregate duplicate Ensembl IDs (mean)
* Align gene order to BulkRNABert’s required `common_gene_id.txt`
* Zero-fill missing genes
* Perform sanity checks

### Output

* `depmap_ensembl_aligned.npy`
* Diagnostic plots:

  * Global TPM distribution
  * Housekeeping gene expression

---

## Lessons Learned During Preprocessing

### 1. Gene identifier consistency is critical

Most complexity came from harmonizing Entrez and Ensembl IDs.
Even small mismatches break compatibility silently.

### 2. Model input assumptions must be respected

BulkRNABert requires:

* Fixed gene order
* Full gene vector
* No missing values

### 3. Silent errors are dangerous

Incorrect gene ordering does not throw runtime errors — but invalidates biological results.

Explicit checks are essential:

* Shape verification
* NaN checks
* Distribution plots

### 4. Zero-filling is a biological assumption

Zero-filling implies:

* “Not expressed” or
* “Not measured”

This choice affects downstream interpretation.

---

# 2️⃣ Embedding Extraction

Script: `scripts/extract_embeddings_depmap.py`

### Input

* `depmap_ensembl_aligned.npy`

### Model

* Pretrained BulkRNABert (TCGA checkpoint)
* No fine-tuning
* Mean pooling of token embeddings from layer 4

### Output

* `embeddings.npy` (n_samples × 256)

---

## HPC & Container Setup

BulkRNABert depends on:

* Specific JAX version
* dm-haiku
* Python 3.11

To ensure reproducibility:

* Built Singularity/Apptainer container (Ubuntu 22.04)
* Installed pinned dependencies
* Used CPU inference with batching
* Used numpy memmap for memory safety

This ensured:

* Stable runtime environment
* Reproducible embeddings
* No cluster-level dependency conflicts

---

# 3️⃣ Downstream Analysis

Evaluations performed:

* PCA visualization
* UMAP visualization
* kNN lineage classification
* Balanced accuracy
* Macro F1 score

Comparison methods:

* BulkRNABert embeddings
* PCA (all genes)
* PCA (HVG 2000)

---

# 📊 Observations

* Raw gene expression (HVG PCA) shows clear tissue separation.
* BulkRNABert embeddings show weaker lineage separation.
* Hematopoietic cancers form a tight cluster in both spaces.

Quantitatively:

* Expression PCA outperforms pretrained embeddings for lineage classification.

---

# Important Technical Reflection

You previously suspected that weaker clustering might be due to:

* Skipping attention layers
* Using mean pooling

Let’s clarify this correctly.

---

## What do attention layers do?

Attention layers:

* Model gene–gene relationships
* Learn co-expression structure
* Capture higher-order transcriptional programs

If attention is removed:

* The model reduces to a shallow embedding
* Biological signal can weaken

However:
If you used the official forward pass with attention enabled (as in your final script), then attention was included.

So weaker separation is unlikely due to skipped attention (if official inference was used).

---

## What does mean pooling do?

Mean pooling:

* Averages token embeddings across all genes
* Treats all genes equally

Biological implication:

* Strong tissue markers get averaged with housekeeping genes
* Fine lineage signal may dilute

This is a real limitation of global pooling.

But:
This is how the model was designed and how embeddings are intended to be extracted.

So this is not an error — it is a design choice.

---

# 🧠 Biological Interpretation

Tissue identity is:

* Strongly encoded in raw gene expression
* Often driven by a small number of marker genes

Pretrained tumor models:

* Learn broader transcriptional programs
* May emphasize global patterns over fine tissue markers

Additionally:

* BulkRNABert was trained on tumors
* DepMap contains cell lines grown in vitro
* Domain mismatch likely affects transfer

---

# 🚀 Next Steps

Instead of debugging architecture:

1. Compare against other foundation models (e.g., Flexynesis)

---

# Final Conclusion

This project establishes a reproducible baseline evaluation of BulkRNABert embeddings on DepMap.

For tissue lineage classification:

* Standard expression PCA remains a strong baseline.
* Pretrained tumor embeddings do not outperform simple expression methods.

This provides a clear starting point for further model comparison and biological exploration.
