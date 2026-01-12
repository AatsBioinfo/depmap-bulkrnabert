DepMap bulk RNA-seq data: DepMap provides high-quality, standardized RNA-seq profiles across a large number of cancer cell lines, spanning multiple tissues and cancer types. This diversity makes DepMap suitable for testing whether pretrained embeddings capture biologically meaningful structure rather than dataset-specific artifacts.
Another important factor was that DepMap data are well documented and widely used, which reduces uncertainty about preprocessing and normalization. Since the goal was to apply an existing RNA-seq transformer model.

Why I chose OmicsExpressionProteinCodingGenesTPMLogp1.csv

After clarifying model input requirements (with the help of ChatGPT), I selected the DepMap file
OmicsExpressionProteinCodingGenesTPMLogp1.csv.

This file was chosen because:

	It contains bulk RNA-seq expression data, not inferred or derived features.
	Expression values are provided as log2(TPM + 1), which is a commonly used normalization and appropriate for neural network inputs.
	It includes protein-coding genes only, which aligns with the gene vocabulary used by BulkRNABert.
	The data are organized as samples × genes, which matches the expected input structure of the model.
	Using this file avoided the need for additional normalization steps and minimized assumptions about expression scaling.

Reading the BulkRNABert GitHub and identifying preprocessing constraints

Before implementing preprocessing, I read through the BulkRNABert section of the multiomics-open-research repository. From the repository structure and documentation, especially: data/bulkrnabert/common_gene_id.txt preprocessing examples used for TCGA data model inference scripts

I identified several non-negotiable input requirements:

	Fixed gene list and ordering
	BulkRNABert expects expression values in the exact gene order defined in common_gene_id.txt.
	Any mismatch in ordering would silently corrupt the input representation.
	Use of Ensembl gene identifiers
	The model’s gene list is defined entirely using Ensembl gene IDs.
	DepMap gene columns use Entrez Gene IDs, embedded in column names.
	Therefore, an explicit Entrez → Ensembl mapping step was required.
	Complete feature vector: The model expects values for all genes in its gene list.
	Genes missing from DepMap must still be represented, which implies zero-filling.

These constraints directly determined how the preprocessing pipeline was designed.

How I started preprocessing the DepMap data
1. Loading and validating the expression matrix
I loaded the DepMap expression file into a pandas DataFrame, set the first column as the sample identifier, and checked the overall shape. This ensured that the data were read correctly and that rows corresponded to samples and columns to genes.

2. Extracting Entrez Gene IDs from column headers
DepMap gene columns are labeled in the format "GENE_NAME (ENTREZ_ID)".
To enable systematic gene mapping, I extracted the numeric Entrez ID from each column name.
This step standardizes gene identifiers and prepares the dataset for merging with external annotation files. I also explicitly checked for non-numeric headers to catch any unexpected formatting issues early.

3. Mapping Entrez IDs to Ensembl IDs
To convert gene identifiers into the format required by BulkRNABert, I used the NCBI gene2ensembl mapping file, filtered to human genes only (tax_id = 9606).
This step:Ensured compatibility with BulkRNABert’s gene list
Made gene identifiers consistent with the model’s training data
Allowed explicit tracking of unmapped genes
Genes that could not be mapped to Ensembl IDs were logged and excluded from downstream steps.

4. Aggregating duplicate Ensembl IDs
During mapping, I observed that multiple Entrez IDs can map to the same Ensembl gene ID.
To avoid duplicate gene entries, I grouped by Ensembl ID and aggregated expression values using the mean.
This choice avoids arbitrarily selecting one mapping and produces a single expression profile per Ensembl gene, which is required for fixed-length model input.

5. Aligning to the BulkRNABert gene list
The processed expression matrix was then aligned to the exact gene list provided in
data/bulkrnabert/common_gene_id.txt.
This involved:
Reindexing genes to match the model’s required order
Identifying genes expected by the model but absent from DepMap
Zero-filling missing genes so the input dimensionality exactly matched the model specification
I explicitly verified that the final gene order exactly matched the provided gene list, as even a single misplaced gene would invalidate downstream embeddings.

6. Final formatting for model input

The final output was stored as a NumPy array with shape:
(number of samples × number of genes)
and saved as:
depmap_ensembl_aligned.npy


Sanity checks and validation

To verify preprocessing correctness, I included multiple sanity checks:
1. Global expression distribution :	
	Histogram of log1p(TPM) values across all genes and samples
	Confirmed expected right-skewed distribution

2. Housekeeping gene expression:
	Examined GAPDH and ACTB distributions to confirm consistent expression

3. Summary statistics:
	Checked min, max, mean, and proportion of zeros

These checks provided confidence that preprocessing preserved biological signal and that no major errors were introduced during alignment.
