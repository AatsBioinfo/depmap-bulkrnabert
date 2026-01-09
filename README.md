# depmap-bulkrnabert

This repository provides a reproducible pipeline to extract and evaluate **BulkRNABert** embeddings from **DepMap bulk RNA-seq** data.

## Overview

We use the pretrained BulkRNABert model to generate gene expression embeddings from DepMap RNA-seq (TPM) data. Downstream analyses include PCA and UMAP to assess biological structure in the embedding space.

## Pipeline Steps

### 1. **Preprocessing**

**Script:** `scripts/preprocess_depmap.py`

* Input: `OmicsExpressionProteinCodingGenesTPMLogp1.csv` from DepMap
* Maps Entrez → Ensembl IDs using NCBI gene2ensembl
* Aggregates duplicate Ensembl IDs by mean
* Aligns to `common_gene_id.txt` from BulkRNABert
* Outputs: `depmap_ensembl_aligned.npy`, diagnostic plots (e.g. `tpm_distribution.png`)

### 2. **Embedding Inference**

**Script:** `scripts/extract_embeddings_depmap.py`

* Loads pretrained BulkRNABert from `multiomics-open-research`
* Inputs: `depmap_ensembl_aligned.npy`
* Runs forward pass in batches
* Outputs: `embeddings.npy` (n_samples × 256)

### 3. **PCA + UMAP Visualization**

**Script:** `scripts/step3_pca_umap.py`

* Inputs: `embeddings.npy` and DepMap metadata
* Runs PCA and UMAP
* Colors by: `TissueOrigin`, `OncotreeLineage`, `OncotreePrimaryDisease`
* Also compares with PCA of raw TPM data
* Outputs: `*.png` plots in `outputs/step3`

## How to Reproduce

```bash
# Clone the repo
$ git clone https://github.com/AatsBioinfo/depmap-bulkrnabert.git
$ cd depmap-bulkrnabert

# Set up environment
$ apptainer exec --bind /fast:/fast container/python311.sif bash -lc "source env/.venv/bin/activate"

# Step 1: Preprocessing
$ python scripts/preprocess_depmap.py

# Step 2: Inference (inside SLURM)
$ sbatch scripts/run_inference.sbatch

# Step 3: PCA/UMAP plots
$ sbatch scripts/run_step3.sbatch
```

## Key Files

| File/Folder  | Description                        |
| ------------ | ---------------------------------- |
| `scripts/`   | All pipeline scripts               |
| `container/` | Singularity def file + built image |
| `outputs/`   | Embeddings and PCA/UMAP figures    |
| `.gitignore` | Tracks exclusions                  |
| `README.md`  | This file                          |

## Dependencies

* Python 3.11
* JAX + Haiku (CPU)
* pandas, numpy, matplotlib, tqdm, scikit-learn, umap-learn

Container is defined in: `container/bulkrnabert.def`

---

## Acknowledgements

This work uses the [BulkRNABert model](https://github.com/EPFL-LCSB/multiomics-open-research) and DepMap transcriptomic data.

---
