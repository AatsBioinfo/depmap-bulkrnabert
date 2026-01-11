depmap-bulkrnabert

This repository provides a reproducible pipeline to extract and evaluate BulkRNABert embeddings from DepMap bulk RNA-seq data.
Overview

We use the pretrained BulkRNABert model to generate gene expression embeddings from DepMap RNA-seq (TPM) data. Downstream analyses include PCA and UMAP to assess biological structure in the embedding space.
Pipeline Steps
1. Preprocessing
- [Lab notes: DepMap preprocessing](prep_LAB_NOTES.md)

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

- [Embedding extraction lab notes](embedding_LAB_NOTES.md)

Script: scripts/extract_embeddings_depmap.py

    Loads pretrained BulkRNABert from multiomics-open-research
    Inputs: depmap_ensembl_aligned.npy
    Runs forward pass in batches
    Outputs: embeddings.npy (n_samples × 256)

Lessons learned

	Pretrained models often require careful adaptation rather than direct application.
	Understanding the forward pass is crucial for efficient inference.
	Not all components of a model are necessary for every task.
	Memory considerations strongly influence practical design choices on HPC systems.


(Environment and container setup (Singularity)
Why I used a container :Initially I ran into a lot of issues with version mismatches and BulkRNABert relies on a specific Python/JAX/Haiku software stack. On an HPC cluster, system Python packages can differ across nodes and change over time. To make the workflow reproducible and easier to run consistently, I used a Singularity container.

The goal of using the container was:
to lock the runtime environment (Ubuntu + Python packages)
to avoid dependency issues on the cluster
to ensure the same code produces the same outputs when rerun

What the container contains (high level)

I built the container using container/bulkrnabert.def with:
Base OS: ubuntu:22.04
Python + build tools
Scientific Python packages: numpy, pandas, matplotlib, scikit-learn, tqdm
BulkRNABert dependencies: jax[cpu] and dm-haiku

The BulkRNABert code itself by copying multiomics-open-research into the image and installing it)
