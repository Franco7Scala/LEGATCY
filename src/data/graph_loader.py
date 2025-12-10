"""
from: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping, raw.pkl, new.pkl)
to: heterodata object + mask_old, mask_new
"""

import os
import torch
import torch_geometric.transforms as T

from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths

from support.utils_data import open_pickle, get_target_type, extract_edge_info
from support.utils import get_base_dir, get_metapaths


def build_heterodata(dataset_name, no_snapshot):
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{no_snapshot}', 'heterodata')

    data = HeteroData()

    # nodes
    for fname in os.listdir(os.path.join(heterodata_dir, 'features')):
        n_type = fname[:-3] #remove the last 4 characters (".pt")
        data[n_type].x = torch.load(heterodata_dir + "/features/" + fname)
        print(f"No of nodes of type {n_type}: {data[n_type].x.shape[0]}")

    #ground truth for target_type
    target_type = get_target_type(dataset_name)
    data[target_type].y = torch.load(os.path.join(heterodata_dir, f'{target_type}_labels.pt'))
    print(f"No of labels: {data[target_type].y.shape[0]}")
    print()

    #edges
    for fname in os.listdir(os.path.join(heterodata_dir, 'edgelists')):
        n_type_src, n_type_tgt, e_type = extract_edge_info(fname)
        edge_index = torch.load(f"{heterodata_dir}/edgelists/{fname}")
        if edge_index.dtype == torch.float64:
            edge_index = edge_index.to(torch.int64)
        data[n_type_src, e_type, n_type_tgt].edge_index = edge_index

    #print("Adding meta-paths...")

    # meta-paths
    metapaths = get_metapaths(dataset_name)
    data = AddMetaPaths(metapaths, weighted=True)(data)

    transform = T.RandomNodeSplit(num_val=0, num_test=0.30) #train-val-test split: 70-0-30
    data = transform(data)

    data.mps = metapaths


    return data


def get_knowledge(dataset_name, no_snapshot, new=True):
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, f"snapshot_{no_snapshot}", "heterodata")
    pickle_name_nodes = f"K_{'new' if new else 'old'}_nodes.pkl"
    pickle_name_edges = f"K_{'new' if new else 'old'}_edges.pkl"
    return open_pickle(os.path.join(heterodata_dir, pickle_name_nodes)), open_pickle(os.path.join(heterodata_dir, pickle_name_edges))


if __name__ == '__main__':
    dataset_name = "openalex"
    NUM_SNAPSHOTS = 7
    for s in range(NUM_SNAPSHOTS):
        data = build_heterodata(dataset_name, s)
        print()
