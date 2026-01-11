depmap-bulkrnabert

This repository provides a reproducible pipeline to extract and evaluate BulkRNABert embeddings from DepMap bulk RNA-seq data.
Overview

We use the pretrained BulkRNABert model to generate gene expression embeddings from DepMap RNA-seq (TPM) data. Downstream analyses include PCA and UMAP to assess biological structure in the embedding space.
Pipeline Steps
1. Preprocessing
- [Lab notes: DepMap preprocessing](LAB_NOTES.md)
Script: scripts/preprocess_depmap.py

    Input: OmicsExpressionProteinCodingGenesTPMLogp1.csv from DepMap
    Maps Entrez → Ensembl IDs using NCBI gene2ensembl
    Aggregates duplicate Ensembl IDs by mean
    Aligns to common_gene_id.txt from BulkRNABert
    Outputs: depmap_ensembl_aligned.npy, diagnostic plots (e.g. tpm_distribution.png)

Lessons Learned During Preprocessing

	1. Gene identifier consistency is critical

	Most preprocessing complexity came from harmonizing Entrez and Ensembl IDs.

	Even small mismatches can completely break model compatibility.

	2. Model input assumptions must be read carefully

	BulkRNABert assumes a fixed gene order and full feature vector.

	These constraints are not obvious unless the repository is examined closely.

	3. Silent errors are the biggest risk

	Incorrect gene ordering or missing genes would not raise runtime errors but would invalidate results.

	Explicit checks (gene order, NaNs, summary statistics) are essential.

	4. Zero-filling is a design choice, not a technical detail

	Filling missing genes with zeros encodes a biological assumption (“not expressed or not measured”).

	This should be documented and justified, not treated as a default operation.

	5. Sanity checks save time later

	Simple plots (global distributions, housekeeping genes) quickly reveal major preprocessing errors.

	These checks increased confidence before running expensive embedding extraction.

2. Embedding Inference

Script: scripts/extract_embeddings_depmap.py

    Loads pretrained BulkRNABert from multiomics-open-research
    Inputs: depmap_ensembl_aligned.npy
    Runs forward pass in batches
    Outputs: embeddings.npy (n_samples × 256)

3. PCA + UMAP Visualization

Script: scripts/step3_pca_umap.py

    Inputs: embeddings.npy and DepMap metadata
    Runs PCA and UMAP
    Colors by: TissueOrigin, OncotreeLineage, OncotreePrimaryDisease
    Also compares with PCA of raw TPM data
    Outputs: *.png plots in outputs/step3

Key Files
File/Folder 	Description
scripts/ 	All pipeline scripts
container/ 	Singularity def file + built image
outputs/ 	Embeddings and PCA/UMAP figures
.gitignore 	Tracks exclusions
README.md 	This file
Dependencies

    Python 3.11
    JAX + Haiku (CPU)
    pandas, numpy, matplotlib, tqdm, scikit-learn, umap-learn

Container is defined in: container/bulkrnabert.def
Acknowledgements
This work uses the BulkRNABert model and DepMap transcriptomic data.
