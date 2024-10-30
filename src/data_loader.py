"""
from: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping, raw.pkl, new.pkl)
to: heterodata object + mask_old, mask_new
"""

import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths

from src.data_utils import open_pickle, get_target_type, get_metapaths



def build_heterodata(dataset_name, no_snapshot):
    heterodata_dir = os.path.join('data', dataset_name, 'snapshot_'+str(no_snapshot), 'heterodata')

    data = HeteroData()

    # nodes
    for fname in os.listdir(os.path.join(heterodata_dir, 'features')):
        n_type = fname[:-3] #remove the last 3 characters (".pt")
        data[n_type].x = torch.load(fname)

    #ground truth for target_type
    target_type = get_target_type(dataset_name)
    data[target_type].y = torch.load(os.path.join(heterodata_dir, f'{target_type}_labels.pt'))

    #edges
    for fname in os.listdir(os.path.join(heterodata_dir, 'edgelists')):
        fname_base = fname[:-3] #remove the last 3 characters (".pt")
        first_underscore = fname_base.find("_") #first occurrence
        last_underscore = fname_base.rfind("_") #last occurrence
        n_type_src = fname_base[:first_underscore]
        e_type = fname_base[first_underscore+1:last_underscore]
        n_type_tgt = fname_base[last_underscore+1:]
        data[n_type_src, e_type, n_type_tgt].edge_index = torch.load(fname)

    #meta-paths
    metapaths = get_metapaths(dataset_name)
    data = AddMetaPaths(metapaths, weighted=True)(data)

    return data



#K_old = open_pickle(os.path.join(heterodata_dir, 'K_old.pkl'))
#K_new = open_pickle(os.path.join(heterodata_dir, 'K_new.pkl'))


