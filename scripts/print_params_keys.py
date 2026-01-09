from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model

CHECKPOINT_DIR = "/fast/home/a/abejoy/data/bulkrnabert/multiomics-open-research/checkpoints"

params, _, _, _ = get_bulkrnabert_pretrained_model(
    model_name="bulk_rna_bert_tcga",
    embeddings_layers_to_save=(4,),
    checkpoint_directory=CHECKPOINT_DIR,
)

print("TOP LEVEL PARAM KEYS:", list(params.keys()))

root = list(params.keys())[0]
print("ROOT:", root)
print("ROOT SUBKEYS:", list(params[root].keys()))

# show deeper keys if present
if "~" in params[root]:
    print("ROOT['~'] SUBKEYS:", list(params[root]["~"].keys()))
    if "expression_embedding" in params[root]["~"]:
        print("expression_embedding keys:", list(params[root]["~"]["expression_embedding"].keys()))

# brute-force search for "expression_embedding" anywhere in tree (shallow)
def walk(d, prefix=""):
    if isinstance(d, dict):
        for k,v in d.items():
            p = f"{prefix}/{k}" if prefix else k
            if "expression" in k:
                print("FOUND KEY:", p)
            walk(v, p)

walk(params)
