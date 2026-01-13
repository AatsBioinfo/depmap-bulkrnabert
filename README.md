depmap-bulkrnabert

This repository provides a reproducible pipeline to extract and evaluate BulkRNABert embeddings from DepMap bulk RNA-seq data (https://github.com/instadeepai/multiomics-open-research).

Overview

We use the pretrained BulkRNABert model to generate gene expression embeddings from DepMap RNA-seq (TPM) data. Downstream analyses include PCA and UMAP to assess biological structure in the embedding space.
Pipeline Steps
1. Preprocessing
- [Lab notes: DepMap preprocessing](prep_LAB_NOTES.md)

Script: scripts/preprocess_depmap.py

- Input: OmicsExpressionProteinCodingGenesTPMLogp1.csv from DepMap
- Maps Entrez → Ensembl IDs using NCBI gene2ensembl
- Aggregates duplicate Ensembl IDs by mean
- Aligns to common_gene_id.txt from BulkRNABert
- Outputs: depmap_ensembl_aligned.npy, diagnostic plots (e.g. tpm_distribution.png)

Lessons Learned During Preprocessing

	1. Gene identifier consistency is critical
	- Most preprocessing complexity came from harmonizing Entrez and Ensembl IDs. Even small mismatches can completely break model compatibility.

	2. Model input assumptions must be read carefully
	- BulkRNABert assumes a fixed gene order and full feature vector.

	3. Silent errors are the biggest risk
	- Incorrect gene ordering or missing genes would not raise runtime errors but would invalidate results. Explicit checks (gene order, NaNs, summary statistics) are essential.

	4. Zero-filling is a design choice, not a technical detail
	- Filling missing genes with zeros encodes a biological assumption (“not expressed or not measured”).

	5. Sanity checks save time later
	- Simple plots (global distributions, housekeeping genes) quickly reveal major preprocessing errors.

2. Embedding Inference

- [Embedding extraction lab notes](embedding_LAB_NOTES.md)

Script: scripts/extract_embeddings_depmap.py

- Loads pretrained BulkRNABert from multiomics-open-research
- Inputs: depmap_ensembl_aligned.npy
- Runs forward pass in batches
- Outputs: embeddings.npy (n_samples × 256)

Lessons learned

	1. Pretrained models often require careful adaptation rather than direct application.
	2. Understanding the forward pass is crucial for efficient inference.
	3. Not all components of a model are necessary for every task.
	4. Memory considerations strongly influence practical design choices on HPC systems.


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
BulkRNABert dependencies: jax[cpu] and dm-haiku. The BulkRNABert code itself by copying multiomics-open-research into the image and installing it)

So as the next step I made graphs for raw tpm pca and depmap pca and also UMAP for depmap - But the clusters do not seem strongly separated. I searched for the cause, my questions were could it be because of the steps I skipped (the attention layers, mean pooling). So what does attention layers and mean pooling does to the data ? 

* Attention is the main part of the transformer that learns gene–gene relationships.
* By skipping it, I likely removed the strongest biological signal.
* Mean pooling across ~19,000 genes weakens marker signal
* Mean pooling treats all genes equally, so tissue-specific genes get averaged together with housekeeping genes and noise - separation becomes weaker.

Maybe lets try : GPU (might not get more OOMs issues)
