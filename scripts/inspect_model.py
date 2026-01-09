import inspect
from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model

CHECKPOINT_DIR = "/fast/home/a/abejoy/data/bulkrnabert/multiomics-open-research/checkpoints"

print("get_bulkrnabert_pretrained_model signature:")
print(inspect.signature(get_bulkrnabert_pretrained_model))

params, forward_fn, tokenizer, config = get_bulkrnabert_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory=CHECKPOINT_DIR,
)

print("\nConfig fields:")
for k in dir(config):
    if not k.startswith("_"):
        try:
            v = getattr(config, k)
            if isinstance(v, (int, float, str, bool, tuple)):
                print(k, "=", v)
        except Exception:
            pass

print("\nforward_fn source (first ~80 lines):")
src = inspect.getsource(forward_fn).splitlines()
print("\n".join(src[:80]))
