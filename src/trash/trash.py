import torch
import pandas as pd
import os

from src.data.data_utils import open_pickle, save_dict_to_pickle
from src.support.utils import get_base_dir



dataset_name = "openalex"
type_nodes = "old"

for i in range(7):
    name = os.path.join(get_base_dir(), f"openalex/snapshot_{i}/heterodata/", f"K_{type_nodes}_nodes.pkl")
    nodes = open_pickle(name)

    for key in nodes.keys():
        nodes[key[:len(key) - 1]] = nodes.pop(key)

    #save_dict_to_pickle(nodes, name)







"""
By FLESCA:
Extracts a full k-hop heterogeneous subgraph from a HeteroData object, keeping the selected nodes and their k-hop neighbors.
Peripheral nodes are included for message passing but excluded from any training/validation/test masks.

Args:
data (HeteroData): Original heterogeneous graph.
selected_nodes_dict (dict): { node_type: Tensor(node_ids) } of core nodes.
target_type (str): Node type that has train/val/test masks.
k (int): Number of hops for neighborhood expansion.

Returns:
HeteroData: Subgraph containing all k-hop neighbors, with core/peripheral masks and adjusted train/val/test masks for target_type.
"""
def extract_hetero_k_hop_subgraph(data, selected_nodes_dict, target_type, k=2):

    loader = HGTLoader(
        data,
        num_samples={ntype: [selected_nodes_dict[ntype].shape[0]] * k for ntype in selected_nodes_dict.keys()},            # full (non-sampled) neighborhood
        input_nodes=(target_type, selected_nodes_dict[target_type].to(torch.long)),   # core nodes per type
        batch_size=len(selected_nodes_dict[target_type]),  # single full batch
    )

    sub_data = next(iter(loader))  # Fetch full subgraph

    # node_mapping = loader.node_sampler.node_mapping #original → subgraph indices
    node_mapping = {ntype: sub_data.n_id_dict[ntype] for ntype in sub_data.node_types}

    for ntype in sub_data.node_types:
        n_nodes = sub_data[ntype].num_nodes
        sub_data[ntype].core_mask = torch.zeros(n_nodes, dtype=torch.bool)

    for ntype, input_nodes in selected_nodes_dict.items():
        if ntype in node_mapping:
            orig_ids = node_mapping[ntype]  # original node IDs in subgraph order
            core_mask = torch.isin(orig_ids, input_nodes.to(orig_ids.device))
            sub_data[ntype].core_mask = core_mask

    for ntype in sub_data.node_types:
        sub_data[ntype].peripheral_mask = ~sub_data[ntype].core_mask

    device = sub_data[target_type].x.device if 'x' in sub_data[target_type] else torch.device('cpu')
    sub_data[target_type].core_mask = sub_data[target_type].core_mask.to(device)

    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in sub_data[target_type]:
            sub_data[target_type][mask_name] = sub_data[target_type][mask_name].to(device)
            sub_data[target_type][mask_name] &= sub_data[target_type].core_mask

    return sub_data, node_mapping















dataset_name = "openalex"
mapping_labels = open_pickle(os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))

for i in range(7):
    df = pd.read_csv(os.path.join(get_base_dir(), f"openalex/snapshot_{i}/original_data/", "author_labels.csv"))
    df['label'] = df['label'].map(mapping_labels)
    Y = torch.tensor(df['label'].values)
    torch.save(Y, os.path.join(get_base_dir(), f"openalex/snapshot_{i}/heterodata/author_labels.pt"))










dataset_name = "openalex"
mapping_labels = open_pickle(os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))
mapping_labels["generic"] = 22

save_dict_to_pickle(mapping_labels, os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))

























path_to_swap_src = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/heterodata/edgelists/author_writes_paper.pt"
path_to_swap_dst = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/heterodata/edgelists/paper_is_written_by_author.pt"

reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)

reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)


edge_index = torch.load(path_to_swap_src)

reversed_edge_index = torch.zeros(edge_index.shape, dtype=edge_index.dtype)
reversed_edge_index[0] = edge_index[1]
reversed_edge_index[1] = edge_index[0]

reversed_edge_index = torch.nan_to_num(reversed_edge_index)

torch.save(reversed_edge_index, path_to_swap_dst)

reversed_edge_index = torch.load(path_to_swap_dst)

print("Finito!")
