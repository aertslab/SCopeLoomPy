import anndata
import loompy as lp
import pandas as pd
from pathlib import Path
import os
from scipy.sparse import issparse
import numpy as np
import json
from scopeloompy import utils

class Loom():
    
    GLOBAL_META_DATA_KEY = 'MetaData'
    
    def __init__(self, file_path, title, hierarchy):
        self.debug = True
        self.file_path = Path(file_path)
        self.title = title
        if len(hierarchy) != 3:
            raise ValueError("The given hierarchy should at least contain 3 empty strings.")
        self.hierarchy = dict(zip(["SCopeTreeL1","SCopeTreeL2","SCopeTreeL3"], hierarchy))
        self.loom_connection = None
        # Global attributes
        self.global_attrs = dict()
        self.meta_data = dict()
        
        # Matrix
        self.matrix = None
        
        # Row attributes
        self.row_attrs = dict()
        self.row_index = None
        
        # Column attributes
        self.col_attrs = dict()
        self.col_index = None
        self.embedding = None
        self.embeddings_x = None
        self.embeddings_y = None
    
    def from_anndata(self, ann_data):
        # Set the row attributes and the row index of this Loom
        if(self.debug):
            print("Setting the row attributes...")
        row_attrs = ann_data.var.to_dict('list')
        row_attrs['var_names'] = ann_data.var_names.values
        self.row_attrs = row_attrs
        self.row_index = ann_data.var_names.values
        # Set the column attributes and the column index of this Loom
        if(self.debug):
            print("Setting the column attributes...")
        col_attrs = ann_data.obs.to_dict('list')
        col_attrs['obs_names'] = ann_data.obs_names.values
        self.col_attrs = col_attrs
        self.col_index = ann_data.obs_names.values
        # Add obsm to col_attrs
        if(self.debug):
            print("Adding obsm to the column attributes...")
        obsm = dict(('{0}'.format(k), ann_data.obsm[k][:,0:2]) for k in ann_data.obsm.keys())
        col_attrs.update(obsm)
        self.matrix = ann_data.X.T
        if issparse(self.matrix):
            self.matrix = self.matrix.tocoo()
        self.init()
        self.finalize()
    
    def from_h5ad(self, file_path):
        if(self.debug):
            print("Reading AnnData file...")
        adata = anndata.read(filename=file_path)
        self.from_anndata(ann_data=adata)
    
    def init(self):
        self.init_meta_data_embeddings()
        self.init_embedding()
        self.init_extra_embeddings()
        self.init_meta_data_annotations()
        self.init_meta_data_metrics()
    
    def update(self):
        '''
        Update all col_attrs, row_attrs and attrs dictionnaries
        '''
        col_attrs = {"Embedding": utils.df_to_named_matrix(self.embedding),
             "Embeddings_X": utils.df_to_named_matrix(self.embeddings_x),
             "Embeddings_Y": utils.df_to_named_matrix(self.embeddings_y),
        }
        self.col_attrs.update(col_attrs)
        row_attrs = {"Gene": self.row_index}
        self.row_attrs.update(row_attrs)
        attrs = {"title": self.title }
        attrs[Loom.GLOBAL_META_DATA_KEY] = self.meta_data
        attrs.update(self.hierarchy)
        self.global_attrs.update(attrs)
    
    def finalize(self):
        '''
        Save this Loom object as a .loom file.
        '''
        if self.debug:
            print("Saving the Loom object as .loom file...")
        if self.file_path.exists():
            self.file_path.unlink()
        # Compress MetaData global attribute
        if Loom.GLOBAL_META_DATA_KEY in self.global_attrs:
            self.global_attrs[Loom.GLOBAL_META_DATA_KEY] = utils.compress_encode(value=self.global_attrs[Loom.GLOBAL_META_DATA_KEY])
        lp.create(os.fspath(path=self.file_path.resolve()), 
                  self.matrix, 
                  row_attrs=self.row_attrs, 
                  col_attrs=self.col_attrs, 
                  file_attrs=self.global_attrs)       

    def init_meta_data_embeddings(self):
        self.meta_data['embeddings'] = list()
        
    def init_embedding(self):
        self.embedding = pd.DataFrame(index=self.col_index)
    
    def init_extra_embeddings(self):
        self.embeddings_x = pd.DataFrame(index=self.col_index)
        self.embeddings_y = pd.DataFrame(index=self.col_index)
    
    def get_nb_embeddings(self):
        if self.embeddings_x.shape[1] == self.embeddings_y.shape[1]:
            return self.embeddings_x.shape[1]-1
        raise ValueError("Dimension between column attributes Embeddings_X and Embeddings_Y do not match.")
    
    def add_embedding_meta_data(self, id, name):
        em = {
            "id": str(id),
            "name": name
        }
        # Filter out all embeddings having the given or the given name
        if len(self.meta_data['embeddings']) > 0:
            self.meta_data['embeddings'] = list(filter(lambda x: x["id"] != id and x["name"] != name, self.meta_data['embeddings']))
        # Add the given embedding
        self.meta_data["embeddings"].append(em)
    
    def add_embedding_by_id(self, id, name, embedding):
        is_default = True if id < 0 else False
        if not is_default:
            id = self.get_nb_embeddings()
        # Add meta data for the given embedding
        self.add_embedding_meta_data(id=id, name=name)
        if is_default:
            # Add embedding as default
            self.embedding["_X"] = embedding[:,0]
            self.embedding["_Y"] = embedding[:,1]
        # Add embedding in Embeddings_{X,Y} column attributes
        self.embeddings_x[str(id)] = embedding[:,0]
        self.embeddings_y[str(id)] = embedding[:,1]
        # Update the all attributes
        self.update()
    
    def add_embedding(self, name, embedding, is_default):
        id = -1 if is_default else len(self.meta_data['embeddings'])
        self.add_embedding_by_id(id=id, name=name, embedding=embedding)
    
    def add_embedding_from_col_attrs(self, key, name, is_default, do_remove = False):
        if self.debug:
            print("Adding embedding {0} from existing column attribute...".format(key))
        ed = self.col_attrs[key]
        self.add_embedding(name=name, embedding=ed, is_default=is_default)
        if do_remove:
            del self.col_attrs[key]
    
    def init_meta_data_annotations(self):
        self.meta_data['annotations'] = list()
    
        
    def add_annotation_meta_data(self, name, values):
        am = {
            "name": name,
            "values": list(map(str, np.unique(values).tolist()))
        }
        # Filter out all embeddings having the given or the given name
        if len(self.meta_data['annotations']) > 0:
            self.meta_data['annotations'] = list(filter(lambda x: x["name"] != name, self.meta_data['annotations']))
        # Add the given embedding
        self.meta_data["annotations"].append(am)
    
    def add_annotation_from_col_attrs(self, key):
        if self.debug:
            print("Adding annotation ({0}) from existing column attribute...".format(key))
        if key not in self.col_attrs:
            raise ValueError("Cannot find annotation with {} as key.".format(key))
        ad = self.col_attrs[key]
        if len(np.unique(ad)) > 245:
            raise ValueError("Cannot add an annotation with more than 245 unique values.")
        self.add_annotation_meta_data(name=key, values=ad)
        # Update all attributes
        self.update()
    
    def init_meta_data_metrics(self):
        self.meta_data['metrics'] = list()

    def add_metric_meta_data(self, name):
        am = {
            "name": name
        }
        # Filter out all embeddings having the given or the given name
        if len(self.meta_data['metrics']) > 0:
            self.meta_data['metrics'] = list(filter(lambda x: x["name"] != name, self.meta_data['metrics']))
        # Add the given embedding
        self.meta_data["metrics"].append(am)

    def add_metric_from_col_attrs(self, key):
        if self.debug:
            print("Adding metric ({0}) from existing column attribute...".format(key))
        if key not in self.col_attrs:
            raise ValueError("Cannot find metric with {0} as key.".format(key))
        self.add_metric_meta_data(name=key)
        # Update all attributes
        self.update()