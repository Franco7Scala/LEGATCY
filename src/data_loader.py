"""
from: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping, raw.pkl, new.pkl)
to: heterodata object + mask_old, mask_new
"""

import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths

from src.data_utils import open_pickle, get_target_type, extract_edge_info, get_metapaths



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
        info = extract_edge_info(fname)
        n_type_src = info[0]
        n_type_tgt = info[1]
        e_type = info[2]
        data[n_type_src, e_type, n_type_tgt].edge_index = torch.load(fname)

    #meta-paths
    metapaths = get_metapaths(dataset_name)
    data = AddMetaPaths(metapaths, weighted=True)(data)

    return data


def get_knowledge(dataset_name, no_snapshot, new=True):
    heterodata_dir = os.path.join('data', dataset_name, 'snapshot_'+str(no_snapshot), 'heterodata')
    if new:
        pickle_name_nodes = 'K_new_nodes.pkl'
        pickle_name_edges = 'K_new_edges.pkl'
    else:
        pickle_name_nodes = 'K_old_nodes.pkl'
        pickle_name_edges = 'K_old_edges.pkl'
    return open_pickle(os.path.join(heterodata_dir, pickle_name_nodes)), open_pickle(os.path.join(heterodata_dir, pickle_name_edges))





