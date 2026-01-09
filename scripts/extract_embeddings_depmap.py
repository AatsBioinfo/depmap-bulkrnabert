import os
import numpy as np
from tqdm import tqdm

import haiku as hk
import jax
import jax.numpy as jnp

from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model
from multiomics_open_research.bulk_rna_bert.model import BulkRNABert

X_NPY = "/fast/home/a/abejoy/data/depmap/processed/depmap_ensembl_aligned.npy"
CHECKPOINT_DIR = "/fast/home/a/abejoy/data/bulkrnabert/multiomics-open-research/checkpoints"
OUT_EMB = "/fast/home/a/abejoy/projects/depmap_bulkrnabert/outputs/embeddings.npy"


def main():
    print("JAX devices:", jax.devices())

    X_log2 = np.load(X_NPY)  # (N, 19062) log2(TPM+1)
    n_samples, n_genes = X_log2.shape
    print("Input shape:", X_log2.shape)

    # Convert log2(TPM+1) -> log10(TPM+1)
    X_log10 = (X_log2 * np.log10(2.0)).astype(np.float32)

    # Load pretrained params/tokenizer/config (params were created with BulkRNABert.__call__)
    params, _orig_forward_fn, tokenizer, config = get_bulkrnabert_pretrained_model(
        model_name="bulk_rna_bert_tcga",
        embeddings_layers_to_save=(4,),  
        checkpoint_directory=CHECKPOINT_DIR,
    )
    assert n_genes == config.n_genes, f"Gene mismatch: X has {n_genes}, model expects {config.n_genes}"

    def forward_skip_attention(tokens: jnp.ndarray) -> jnp.ndarray:
        """
        Call BulkRNABert exactly like pretrained forward, but:
        - bypass attention blocks (OOM source)
        - bypass lm_head (unneeded + heavy)
        - return pooled pre-attention embedding (B, D)
        """
        model = BulkRNABert(config=config, name="bulk_bert")

        # Monkey-patch attention blocks: identity + stash pre-attn x
        def _no_attention_blocks(x, outs, attention_mask):
            outs["pre_attn"] = x
            return x, outs

        # Monkey-patch lm_head: tiny dummy to avoid computing logits over all genes
        def _dummy_lm_head(x):
            return {"logits": jnp.zeros((x.shape[0], 1), dtype=x.dtype)}

        model.apply_attention_blocks = _no_attention_blocks  # type: ignore
        model._lm_head = _dummy_lm_head  # type: ignore

        # Provide a tiny attention mask to prevent BulkRNABert.__call__ from creating (B,1,L,L)
        b, l = tokens.shape
        small_mask = jnp.ones((b, 1, 1, 1), dtype=jnp.float32)

        outs = model(tokens=tokens, attention_mask=small_mask)
        x = outs["pre_attn"]                # (B, L, D)
        return jnp.mean(x, axis=1)          # (B, D)

    forward = hk.transform(forward_skip_attention)
    rng = jax.random.PRNGKey(0)

    batch_size = 64  # safe now; adjust later
    embs = []

    @jax.jit
    def run_batch(ids, key):
        return forward.apply(params, key, ids)  # (B, 256)

    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        ids = tokenizer.batch_tokenize(X_log10[start:end, :])
        ids = jnp.asarray(ids, dtype=jnp.int32)

        rng, k = jax.random.split(rng)
        embs.append(np.array(run_batch(ids, k)))

    E = np.vstack(embs).astype(np.float32)
    print("Embeddings shape:", E.shape)

    os.makedirs(os.path.dirname(OUT_EMB), exist_ok=True)
    np.save(OUT_EMB, E)
    print("Saved:", OUT_EMB)


if __name__ == "__main__":
    main()
