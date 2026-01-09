import inspect
from multiomics_open_research.bulk_rna_bert import model as m

print("BulkRNABert members that look relevant:")
for name in dir(m.BulkRNABert):
    if any(x in name.lower() for x in ["embed", "token", "encode", "gene", "forward", "__call__"]):
        print(" -", name)

print("\nSource of BulkRNABert.__call__ (first 120 lines):")
src = inspect.getsource(m.BulkRNABert.__call__).splitlines()
print("\n".join(src[:120]))
