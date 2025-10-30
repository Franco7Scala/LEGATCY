import copy
import torch

from torch_geometric.loader import HGTLoader
from src.data.data_utils import create_nodes_dict_empty


def k_hop_subgraph(data, target_type, seeds_mask, k):
    subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        if dst_type == target_type:
            for idx, node in enumerate(data[edge_type]["edge_index"][1]):
                if node.item() in seeds_mask:
                    subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

        if src_type == target_type:
            for idx, node in enumerate(data[edge_type]["edge_index"][0]):
                if node.item() in seeds_mask:
                    subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

    #TODO add meta paths for heterogeneous graphs
    subgraph_mask[target_type] = seeds_mask.tolist()
    subgraph_mask = _merge_masks(subgraph_mask, _hop_traveling(data, target_type, subgraph_mask, k-1))
    for ntype in subgraph_mask.keys():
        subgraph_mask[ntype] = torch.tensor(subgraph_mask[ntype]).to(data[data.node_types[0]].x.device)

    subgraph_data = data.subgraph(subgraph_mask)
    return subgraph_data, subgraph_mask


def _hop_traveling(data, target_type, subgraph_mask, k):
    if k == 0:
        return subgraph_mask

    next_hop_subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        for node_type in subgraph_mask.keys():
            if node_type != target_type:
                prevoius_hop_nodes = subgraph_mask[node_type]
                if dst_type == node_type and src_type != target_type:
                    for idx, node in enumerate(data[edge_type]["edge_index"][1]):
                        if node.item() in prevoius_hop_nodes:
                            next_hop_subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

                if src_type == node_type and dst_type != target_type:
                    for idx, node in enumerate(data[edge_type]["edge_index"][0]):
                        if node.item() in prevoius_hop_nodes:
                            next_hop_subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

    return _hop_traveling(data, target_type, next_hop_subgraph_mask, k-1)


def _merge_masks(first_mask, second_mask):
    merged_mask = {}
    for ntype in first_mask.keys():
        merged_mask[ntype] = []

    for ntype in second_mask.keys():
        merged_mask[ntype] = []

    for ntype in merged_mask.keys():
        mask_1 = first_mask[ntype] if ntype in first_mask else []
        mask_2 = second_mask[ntype] if ntype in second_mask else []
        merged_mask[ntype] = list(set(mask_1 + mask_2))

    return merged_mask
