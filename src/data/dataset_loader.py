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


def load_dataset(dataset_name, metapaths_enabled, n_snapshot, times_fist_snapshot, k, device, percentage_test_set=0.2):
    path = os.path.join(get_base_dir(), dataset_name)
    snapshot_masks = [] # list of masks (train and test) for each snapshot (for heterogeneous graph)

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
        if "x" not in dataset.data[ntype]:
            num_nodes = dataset.data[ntype].num_nodes
            dataset.data[ntype].x = torch.zeros((num_nodes, in_dim), device=device)

    metapaths = get_metapaths(dataset_name)
    dataset.data.mps = metapaths
    if metapaths_enabled:
        dataset.data = AddMetaPaths(metapaths=metapaths, weighted=True)(dataset.data)

    dataset.data.target_type = target_type
    dataset.data.to(device)
    dataset.data.device = dataset.data.x_dict[dataset.data.target_type].device
    # determining snapshot masks
    size = dataset.data[target_type].x.shape[0]
    seed_mask = torch.randperm(size)
    # first snapshot
    n_samples_first_snapshot = int((size / (n_snapshot + times_fist_snapshot)) * times_fist_snapshot)
    train_mask = seed_mask[0: int(n_samples_first_snapshot * (1 - percentage_test_set))]
    test_mask = seed_mask[int(n_samples_first_snapshot * (1 - percentage_test_set)): n_samples_first_snapshot]
    snapshot_masks.append({"train": k_hop_subgraph(dataset.data, train_mask, k)[1], "test": k_hop_subgraph(dataset.data, test_mask, k)[1]})
    # remaining snapshots
    splitting_size = size - n_samples_first_snapshot
    split_size = int(splitting_size / n_snapshot)
    for i in range(n_snapshot):
        train_mask = seed_mask[int(i * split_size) + n_samples_first_snapshot: int(((i + 1) * split_size) - (split_size * (percentage_test_set)) + n_samples_first_snapshot)]
        test_mask = seed_mask[int(((i + 1) * split_size) - (split_size * (percentage_test_set)) + n_samples_first_snapshot): int((i + 1) * split_size) + n_samples_first_snapshot]
        snapshot_masks.append({"train": k_hop_subgraph(dataset.data, train_mask, k)[1], "test": k_hop_subgraph(dataset.data, test_mask, k)[1]})

    return dataset.data, target_type, snapshot_masks
