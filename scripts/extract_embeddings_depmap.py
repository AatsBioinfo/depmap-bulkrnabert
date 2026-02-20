#!/usr/bin/env python3

import os

# Must be set before importing jax
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from pathlib import Path
import numpy as np
from tqdm import tqdm

import haiku as hk
import jax
import jax.numpy as jnp

from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model


# -----------------------
# Paths
# -----------------------
X_NPY = "/fast/home/a/abejoy/data/depmap/processed/depmap_ensembl_aligned.npy"
CHECKPOINT_DIR = "/fast/home/a/abejoy/data/bulkrnabert/multiomics-open-research/checkpoints"

OUT_DIR = "/fast/home/a/abejoy/projects/depmap_bulkrnabert/outputs"
OUT_DAT = os.path.join(OUT_DIR, "embeddings.dat")
OUT_NPY = os.path.join(OUT_DIR, "embeddings.npy")

# -----------------------
# Runtime knobs
# -----------------------
BATCH_SIZE = int(os.environ.get("BULKRNABERT_BATCH_SIZE", "1"))  # start at 1; increase cautiously
USE_JIT = os.environ.get("BULKRNABERT_USE_JIT", "0") == "1"      # default off (JIT can spike RAM)


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("JAX devices:", jax.devices())
    print("Batch size:", BATCH_SIZE, "| JIT:", USE_JIT)

    # Memmap input (stream from disk)
    X_log2 = np.load(X_NPY, mmap_mode="r")  # (N, 19062) log2(TPM+1)
    n_samples, n_genes = X_log2.shape
    print("Input:", X_NPY)
    print("Input shape:", X_log2.shape, "dtype:", X_log2.dtype)

    # Load pretrained model/tokenizer/config (OFFICIAL API)
    params, forward_fn, tokenizer, config = get_bulkrnabert_pretrained_model(
        model_name="bulk_rna_bert_tcga",
        embeddings_layers_to_save=(4,),
        checkpoint_directory=CHECKPOINT_DIR,
    )
    forward = hk.transform(forward_fn)

    assert n_genes == config.n_genes, f"Gene mismatch: X has {n_genes}, model expects {config.n_genes}"

    rng = jax.random.PRNGKey(0)

    # Apply wrapper (optionally JIT)
    if USE_JIT:
        @jax.jit
        def apply_model(p, key, ids):
            return forward.apply(p, key, ids)
    else:
        def apply_model(p, key, ids):
            return forward.apply(p, key, ids)

    # We allocate output after first batch to infer embedding dim from actual model output
    E = None
    emb_dim = None

    embedding_key = "embeddings_4"  # per official README example

    for start in tqdm(range(0, n_samples, BATCH_SIZE), desc="Embedding", unit="batch"):
        end = min(start + BATCH_SIZE, n_samples)

        # Load batch into RAM
        x_batch_log2 = np.asarray(X_log2[start:end, :], dtype=np.float32)

        # Convert log2(TPM+1) -> log10(TPM+1) (what official examples use)
        x_batch_log10 = x_batch_log2 * np.log10(2.0)

        # Tokenize per batch
        ids = tokenizer.batch_tokenize(x_batch_log10)  # (B, 19062) ints
        ids = jnp.asarray(ids, dtype=jnp.int32)

        rng, k = jax.random.split(rng)
        outs = apply_model(params, k, ids)

        if embedding_key not in outs:
            raise KeyError(f"Expected key '{embedding_key}' in outs. Keys: {list(outs.keys())}")

        token_emb = outs[embedding_key]          # (B, 19062, D)
        emb = jnp.mean(token_emb, axis=1)        # (B, D)
        emb_np = np.array(emb, dtype=np.float32)

        # Allocate disk-backed output once we know D
        if E is None:
            emb_dim = emb_np.shape[1]
            print("Detected embedding dim from model output:", emb_dim)
            E = np.memmap(OUT_DAT, mode="w+", dtype="float32", shape=(n_samples, emb_dim))

        # Write and flush occasionally
        E[start:end, :] = emb_np
        if start % (50 * BATCH_SIZE) == 0:
            E.flush()
            print(f"[progress] wrote rows {start}..{end-1} / {n_samples}")

    assert E is not None and emb_dim is not None
    E.flush()

    # Save .npy for convenience
    np.save(OUT_NPY, np.array(E))
    print("Saved embeddings:")
    print("  ", OUT_DAT, f"(memmap, shape=({n_samples},{emb_dim}))")
    print("  ", OUT_NPY, f"(npy, shape=({n_samples},{emb_dim}))")


if __name__ == "__main__":
    main()
