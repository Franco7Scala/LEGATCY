import copy

import torch
from torch_geometric.loader import HGTLoader

from src.data.data_utils import create_nodes_dict_empty

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


def k_hop_subgraph(data, seeds_mask, k=2):
    target_type = list(seeds_mask.keys())[0]
    seeds_mask = seeds_mask[target_type]
    subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        if dst_type == target_type:
            for idx, node in enumaerate(data[edge_type]["edge_index"][1]):
                if node.item() in seeds_mask:
                    subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

        if src_type == target_type:
            for idx, node in enumaerate(data[edge_type]["edge_index"][0]):
                if node.item() in seeds_mask:
                    subgraph_mask[src_type].append(data[edge_type]["edge_index"][1][idx].item())

    #TODO add meta paths for heterogeneous graphs
    return data.subgraph(_merge_masks(seeds_mask, _hop_traveling(data, target_type, subgraph_mask, k-1)))


def _hop_traveling(data, target_type, subgraph_mask, k):
    if k == 0:
        return subgraph_mask

    next_hop_subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        for node_type in subgraph_mask.keys():
            prevoius_hop_nodes = subgraph_mask[node_type]
            if dst_type == node_type and src_type != target_type:
                for idx, node in enumaerate(data[edge_type]["edge_index"][0]):
                    if node.item() in prevoius_hop_nodes:
                        next_hop_subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

            if src_type == node_type and src_type != target_type:
                for idx, node in enumaerate(data[edge_type]["edge_index"][1]):
                    if node.item() in prevoius_hop_nodes:
                        next_hop_subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

    return _hop_traveling(data, target_type, next_hop_subgraph_mask, k-1)


def _merge_masks(mask1, mask2):
    mask1 = copy.deepcopy(mask1)
    for ntype in mask1.keys():
        mask1[ntype] = list(set(mask1[ntype].extend(mask2[ntype])))

    return mask1
