import torch
from torch_geometric.loader import HGTLoader

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


def extract_hetero_k_hop_subgraph_all(data, selected_nodes_dict, target_type, k=2):


    # Run HGTLoader only from the target_type (PyG requirement)
    loader = HGTLoader(
        data,
        num_samples={ntype: [999999] * k for ntype in data.node_types},  # full neighborhood up to k hops
        input_nodes=(target_type, selected_nodes_dict[target_type].to(torch.long)),
        batch_size=len(selected_nodes_dict[target_type]),
        shuffle=False,
    )

    sub_data = next(iter(loader))
    node_mapping = {ntype: sub_data.n_id_dict[ntype] for ntype in sub_data.node_types}

    # Add any missing explicitly selected nodes (other types)
    for ntype, selected_ids in selected_nodes_dict.items():
        if ntype == target_type:
            continue  # already covered
        if ntype not in sub_data.node_types:
            # This node type was not sampled at all -> add placeholder type
            sub_data[ntype].x = data[ntype].x[selected_ids]
            node_mapping[ntype] = selected_ids
        else:
            # Append any missing node IDs that weren't reached by sampling
            existing_ids = node_mapping[ntype].cpu()
            missing_ids = torch.tensor([nid for nid in selected_ids.tolist() if nid not in existing_ids.tolist()],
                                       dtype=torch.long)
            if len(missing_ids) > 0:
                extra_x = data[ntype].x[missing_ids]
                sub_data[ntype].x = torch.cat([sub_data[ntype].x, extra_x], dim=0)
                node_mapping[ntype] = torch.cat([existing_ids, missing_ids])

    # Add masks
    for ntype in sub_data.node_types:
        n_nodes = sub_data[ntype].num_nodes
        device = sub_data[ntype].x.device if 'x' in sub_data[ntype] else torch.device('cpu')
        sub_data[ntype].core_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)

    for ntype, input_nodes in selected_nodes_dict.items():
        if ntype in node_mapping:
            orig_ids = node_mapping[ntype]
            core_mask = torch.isin(orig_ids, input_nodes.to(orig_ids.device))
            sub_data[ntype].core_mask = core_mask

    for ntype in sub_data.node_types:
        sub_data[ntype].peripheral_mask = ~sub_data[ntype].core_mask

    # Align train/val/test masks (target type only)
    device = sub_data[target_type].x.device if 'x' in sub_data[target_type] else torch.device('cpu')
    sub_data[target_type].core_mask = sub_data[target_type].core_mask.to(device)
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in sub_data[target_type]:
            sub_data[target_type][mask_name] = sub_data[target_type][mask_name].to(device)
            sub_data[target_type][mask_name] &= sub_data[target_type].core_mask

    return sub_data, node_mapping
