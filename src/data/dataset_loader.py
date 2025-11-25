import os
import torch
import torch.nn as nn

from torch_geometric.datasets import IMDB
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.transforms import AddMetaPaths
from src.data.data_utils import create_nodes_dict_empty
from src.support.utils import get_base_dir
from src.support.utils_graph import k_hop_subgraph
from support.utils import get_metapaths


def load_dataset(dataset_name, metapaths_enabled, n_snapshot, times_fist_snapshot, k, device):
    path = os.path.join(get_base_dir(), dataset_name)
    snapshot_masks = []

    if dataset_name.lower() == "imdb".lower():
        dataset = IMDB(path)
        target_type = "movie"

    elif dataset_name.lower() == "dblp".lower():
        dataset = DBLP(path)
        target_type = "author"

    else:
        raise Exception(f"Unknown dataset '{dataset_name}'!")

    in_dim = 128
    embeddings = nn.ModuleDict()
    for ntype in dataset.data.metadata()[0]:  # metadata()[0] returns node types list
        if 'x' not in dataset.data[ntype]:

            num_nodes = dataset.data[ntype].num_nodes
            dataset.data[ntype].x = torch.zeros((num_nodes, in_dim), device=device)

    metapaths = get_metapaths(dataset_name)
    dataset.data.mps = metapaths
    if metapaths_enabled:
        dataset.data = AddMetaPaths(metapaths=metapaths, weighted=True)(dataset.data)

    dataset.data.target_type = target_type
    dataset.data.to(device)
    dataset.data.device = dataset.data.x_dict[dataset.data.target_type].device
    size = dataset.data[target_type].x.shape[0]
    # first snapshot
    n_samples_first_snapshot = int((size / (n_snapshot + times_fist_snapshot)) * times_fist_snapshot)
    mask = torch.arange(0, size)
    mask = mask[0: n_samples_first_snapshot]
    snapshot_masks.append(k_hop_subgraph(dataset.data, mask, k)[1])
    # remaining snapshots
    splitting_size = size - n_samples_first_snapshot
    for i in range(n_snapshot):
        mask = torch.arange(0, size)
        mask = mask[int(i * splitting_size / n_snapshot) + n_samples_first_snapshot: int((i + 1) * splitting_size / n_snapshot) + n_samples_first_snapshot]
        snapshot_masks.append(k_hop_subgraph(dataset.data, mask, k)[1])

    return dataset.data, target_type, snapshot_masks
