# From SCANPY to a Loom Compatible with SCope

Unfortunately, `write_loom` function from `SCANPY` does not store everything inside the loom (e.g.: embeddings.). The following tutorial will guide you to create a .loom file from AnnData object generated by SCANPY that is filcompatible with SCope.

Import `scopeloompy` package:
```
from scopeloompy import *
```

Read a h5ad file generated from SCANPY or read SCANPY AnnData object:
```
loom = Loom(file_path="Anndata.loom", title="Test", hierarchy=["","",""])
# from a .h5ad file
loom.from_h5ad(file_path="~/Test/Anndata.h5ad")
# from AnnData object
loom.from_anndata(ann_data=ann_data)
```

Add embeddings:
```
loom.add_embedding_from_col_attrs(key="X_pca", name="PCA", is_default=True)
loom.add_embedding_from_col_attrs(key="X_umap", name="UMAP", is_default=False)
```

Add annotations (discrete variables):
```
loom.add_annotation_from_col_attrs(key="louvain")
```

Add metrics (continuous variables):
```
loom.add_metric_from_col_attrs(key="n_counts_all")
```

Find all the variables that you can add to the `loom`:
```
loom.col_attrs.keys()
```

Save to .loom file:
```
loom.finalize()
```