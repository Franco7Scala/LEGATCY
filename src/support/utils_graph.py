import copy
import torch

from torch_geometric.loader import HGTLoader
from src.data.data_utils import create_nodes_dict_empty


def k_hop_subgraph(data, seeds_mask, k):
    subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        if dst_type == data.target_type:
            for idx, node in enumerate(data[edge_type]["edge_index"][1]):
                if node.item() in seeds_mask:
                    subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

        if src_type == data.target_type:
            for idx, node in enumerate(data[edge_type]["edge_index"][0]):
                if node.item() in seeds_mask:
                    subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

    #TODO add meta paths for heterogeneous graphs
    subgraph_mask[data.target_type] = seeds_mask.tolist()
    subgraph_mask = _merge_masks(subgraph_mask, _hop_traveling(data, subgraph_mask, k-1))
    for ntype in subgraph_mask.keys():
        subgraph_mask[ntype] = torch.tensor(subgraph_mask[ntype]).to(int).to(data.device)

    subgraph_data = data.subgraph(subgraph_mask)
    return subgraph_data, subgraph_mask


def _hop_traveling(data, subgraph_mask, k):
    if k == 0:
        return subgraph_mask

    next_hop_subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        for node_type in subgraph_mask.keys():
            if node_type != data.target_type:
                prevoius_hop_nodes = subgraph_mask[node_type]
                if dst_type == node_type and src_type != data.target_type:
                    for idx, node in enumerate(data[edge_type]["edge_index"][1]):
                        if node.item() in prevoius_hop_nodes:
                            next_hop_subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

                if src_type == node_type and dst_type != data.target_type:
                    for idx, node in enumerate(data[edge_type]["edge_index"][0]):
                        if node.item() in prevoius_hop_nodes:
                            next_hop_subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

    return _hop_traveling(data, next_hop_subgraph_mask, k-1)


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


def extract_edges(data, nodes):
    edges = {}
    for etype in data.edge_types:
        edges[etype] = []
        src_type, _, dst_type = etype
        for edge in data[etype].edge_index.T:
            src = edge[0]
            dst = edge[1]
            if src in nodes[src_type].to(data.device) or dst in nodes[dst_type].to(data.device):
                edges[etype].append((src, dst))

        edges[etype] = torch.tensor(edges[etype]).to(data.device)

    return edges


def extract_train_data(data, mask):
    # keeping only train nodes
    train_mask = torch.arange(0, data[data.target_type].x.shape[0]).to(data.device)[data[data.target_type].train_mask]
    # keeping only nodes in the train mask
    to_keep_mask = torch.isin(train_mask, torch.tensor(mask[data.target_type]).to(data.device)).reshape(-1)
    mask[data.target_type] = train_mask[to_keep_mask]
    return mask


# getting evaluation data that does not include future nodes, intersection between test_mask and "new_nodes + old_nodes"
def extract_evaluation_data(data, new_nodes, old_nodes, subgraph_hops):
    # keeping only test nodes
    test_mask = torch.arange(0, data[data.target_type].x.shape[0]).to(data.device)[data[data.target_type].test_mask]
    # keeping only new and old nodes
    mask = torch.isin(test_mask, torch.tensor(new_nodes[data.target_type]).to(data.device)) | torch.isin(test_mask, torch.tensor(old_nodes[data.target_type]).to(data.device))
    test_mask = test_mask[mask]
    return k_hop_subgraph(data, test_mask, subgraph_hops)[0].to(data.device)
