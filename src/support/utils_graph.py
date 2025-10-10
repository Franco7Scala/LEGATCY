import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import AddMetaPaths, RandomNodeSplit

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

    loader = NeighborLoader(
        data,
        num_neighbors=[-1] * k,            # full (non-sampled) neighborhood
        input_nodes=selected_nodes_dict,   # core nodes per type
        batch_size=sum(len(v) for v in selected_nodes_dict.values()),  # single full batch
    )

    sub_data = next(iter(loader))  # Fetch full subgraph

    for ntype in sub_data.node_types:
        n_nodes = sub_data[ntype].num_nodes
        sub_data[ntype].core_mask = torch.zeros(n_nodes, dtype=torch.bool)

    node_mapping = loader.node_sampler.node_mapping #original → subgraph indices

    for ntype, input_nodes in selected_nodes_dict.items():
        if ntype in sub_data.node_types and ntype in node_mapping:
            mapping = node_mapping[ntype]
            valid_nodes = input_nodes[input_nodes < len(mapping)]
            sub_indices = mapping[valid_nodes]
            sub_data[ntype].core_mask[sub_indices] = True

    for ntype in sub_data.node_types:
        sub_data[ntype].peripheral_mask = ~sub_data[ntype].core_mask

    sub_data = AddMetaPaths(data.mps, weighted=True)(sub_data)

    #split_transform = RandomNodeSplit(num_val=num_val, num_test=num_test)
    #sub_data = split_transform(sub_data)


    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in sub_data[target_type]:
            sub_data[target_type][mask_name] &= sub_data[target_type].core_mask

    return sub_data
