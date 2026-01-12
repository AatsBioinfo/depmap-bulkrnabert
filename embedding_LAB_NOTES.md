Lab Notes – Embedding Extraction with BulkRNABert: I did not aim to retrain or finetune the model; instead, I treated BulkRNABert as a feature extractor.

How I approached the problem

I started from the assumption that BulkRNABert had already learned biologically meaningful gene-expression patterns during pretraining. Therefore, the main task was to adapt my DepMap data to the model’s expectations and extract representations in a computationally feasible way on an HPC cluster.

The key questions I needed to answer were:

1. What input format does the model expect?
2. Where in the model can I extract a useful representation?

Understanding the pretrained model before running inference: Before running the model, I inspected the BulkRNABert code and pretrained loading functions.Using small inspection scripts, I examined:

*the configuration object returned by get_bulkrnabert_pretrained_model
*the expected number of genes
*the embedding dimension
*the structure of the forward pass (BulkRNABert.__call__)

This step was necessary because the repository does not provide a single “get_embeddings” function, and understanding the forward computation was required to decide which outputs are relevant and which computations can be skipped.

Input preparation:

The preprocessed DepMap expression matrix contains values in log2(TPM + 1) format. Inspection of the model configuration showed that BulkRNABert applies internal log normalization and max normalization before discretizing expression values into bins. Because tokenization is sensitive to the numerical scale of its inputs, I applied a linear rescaling from log2(TPM + 1) to log10(TPM + 1) to reduce potential scale mismatch with the distribution used during tokenizer calibration. This transformation preserves relative expression differences while adjusting the numeric range.”
To ensure compatibility with the pretrained model, I converted the expression values from log2 to log10 scale using a simple linear transformation. This step preserves relative expression differences while matching the numerical scale used during pretraining.

Tokenization of expression values: BulkRNABert does not operate directly on continuous expression values. Instead, expression values are discretized into tokens using a tokenizer.For each batch of samples, I applied the provided tokenizer to convert the expression matrix into integer token IDs. This step is mandatory, as the transformer operates on tokenized inputs rather than raw floating-point values.

Why I modified the forward pass (mainly OOM issues, CPU env): When examining the model’s forward pass, I observed that the default inference path performs full self-attention over all genes, which scales quadratically with the number of genes (~19,000). This resulted in memory issues when running on CPU.
Since my goal was embedding extraction only, and not prediction or reconstruction, I reasoned that: I do not need attention weights and output logits over genes. Therefore, I modified the forward pass to bypass: the attention blocks, and the language-model head. This significantly reduced memory usage while still allowing me to extract meaningful intermediate representations.

Extracting sample-level embeddings

Within the modified forward pass, I captured the pre-attention gene representations produced by the embedding layer.
At this stage, the model outputs a tensor of shape:
(batch_size, number_of_genes, embedding_dimension)


To obtain a single vector per sample, I applied mean pooling across the gene dimension, resulting in:
(batch_size, embedding_dimension)
Mean pooling was chosen as a simple and interpretable way to aggregate gene-level information into a sample-level representation.

Batch inference and computational constraints

To ensure stable execution on the HPC cluster:

*I processed samples in fixed-size batches
*I used JAX’s jit compilation for performance
*I carefully controlled random number generation for reproducibility

Batching was essential to prevent memory exhaustion and ensure reliable execution.

Outputs:

The final output of this step is a NumPy array:
embeddings.npy
with shape:
(number_of_samples, 256)
Each row represents a DepMap cell line as a 256-dimensional embedding vector, which is used directly for downstream dimensionality-reduction analyses.

Difficulties encountered

1. The model does not provide a direct interface for embedding extraction.
2. Full attention over genes is prohibitively expensive on CPU.
3. Understandng the foward pass and structures.
4. Understanding the internal structure of the model required reading and inspecting source code.
