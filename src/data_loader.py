"""
from: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping, raw.pkl, new.pkl)
to: heterodata object + mask_old, mask_new
"""

import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths

from src.data_utils import open_pickle, get_target_type, extract_edge_info, get_metapaths, get_base_dir


def build_heterodata(dataset_name, no_snapshot):
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{no_snapshot}', 'heterodata')

    data = HeteroData()

    # nodes
    for fname in os.listdir(os.path.join(heterodata_dir, 'features')):
        n_type = fname[:-4] #remove the last 4 characters ("s.pt")
        data[n_type].x = torch.load(heterodata_dir + "/features/" + fname)
        print(f"No of nodes of type {n_type}: {data[n_type].x.shape[0]}")

    #ground truth for target_type
    target_type = get_target_type(dataset_name)
    data[target_type].y = torch.load(os.path.join(heterodata_dir, f'{target_type}_labels.pt'))
    print(f"No of nodes of labels: {data[target_type].y.shape[0]}")

    #edges
    for fname in os.listdir(os.path.join(heterodata_dir, 'edgelists')):
        info = extract_edge_info(fname)
        n_type_src = info[0]
        n_type_tgt = info[1]
        e_type = info[2]
        edge_index = torch.load(heterodata_dir + "/edgelists/" + fname)
        if edge_index.dtype == torch.float64:
            edge_index = edge_index.to(torch.int64)
        data[n_type_src, e_type, n_type_tgt].edge_index = edge_index

        src_nodes, tgt_nodes = edge_index[0], edge_index[1]

        # Get the number of nodes for each node type
        num_src_nodes = data[n_type_src].x.shape[0]
        num_tgt_nodes = data[n_type_tgt].x.shape[0]

        # Ensure that all node indices are within valid bounds
        if not (src_nodes < num_src_nodes).all() or not (tgt_nodes < num_src_nodes).all():
            print(f"Warning: Some edge indices are out of range for the respective node types.")
        else:
            print(f"All edge indices are within valid ranges for {info}.")

    #print("Adding meta-paths...")

    #meta-paths
    #metapaths = get_metapaths(dataset_name)
    #data = AddMetaPaths(metapaths, weighted=True)(data)


    return data


def get_knowledge(dataset_name, no_snapshot, new=True):
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{no_snapshot}', 'heterodata')
    if new:
        pickle_name_nodes = 'K_new_nodes.pkl'
        pickle_name_edges = 'K_new_edges.pkl'
    else:
        pickle_name_nodes = 'K_old_nodes.pkl'
        pickle_name_edges = 'K_old_edges.pkl'
    return open_pickle(os.path.join(heterodata_dir, pickle_name_nodes)), open_pickle(os.path.join(heterodata_dir, pickle_name_edges))





