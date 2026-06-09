import os
import torch
import torch.nn as nn

from torch_geometric.datasets import IMDB, AMiner
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.transforms import AddMetaPaths
from support.utils_graph import k_hop_subgraph
from support.utils import get_metapaths, get_base_dir, build_heterodata


def load_dataset(dataset_name, metapaths_enabled, n_snapshot, times_fist_snapshot, k, device, percentage_test_set=0.2):
    path = os.path.join(get_base_dir(), dataset_name)
    snapshot_masks = [] # list of masks (train and test) for each snapshot (for heterogeneous graph)

    if dataset_name.lower() == "imdb".lower():
        target_type = "movie"
        dataset = IMDB(path)
        data = dataset.data

    elif dataset_name.lower() == "dblp".lower():
        target_type = "author"
        dataset = DBLP(path)
        data = dataset.data

    elif dataset_name.lower() == "aminer".lower():
        target_type = "author"
        dataset = AMiner(path)
        data = dataset.data

    elif dataset_name.lower() == "politifact".lower():
        target_type = "news"
        data = build_heterodata(dataset_name, target_type)

    elif dataset_name.lower() == "mumin".lower():
        target_type = "claim"
        data = build_heterodata(dataset_name, target_type)

    else:
        raise Exception(f"Unknown dataset '{dataset_name}'!")

    in_dim = 128
    embeddings = nn.ModuleDict()
    for ntype in dataset.data.metadata()[0]:  # metadata()[0] returns node types list
        if "x" not in dataset.data[ntype]:
            num_nodes = dataset.data[ntype].num_nodes
            dataset.data[ntype].x = torch.zeros((num_nodes, in_dim), device=device)

    metapaths = get_metapaths(dataset_name)
    data.mps = metapaths
    if metapaths_enabled:
        data = AddMetaPaths(metapaths=metapaths, weighted=True)(dataset.data)

    data.target_type = target_type
    data.to(device)
    data.device = data.x_dict[data.target_type].device
    # determining snapshot masks
    size = data[target_type].x.shape[0]
    n_samples_first_snapshot = int((size / (n_snapshot + times_fist_snapshot)) * times_fist_snapshot)
    seed_mask = _fixed_randperm(size, n_samples_first_snapshot).to(device)
    # first snapshot
    train_mask = seed_mask[0: int(n_samples_first_snapshot * (1 - percentage_test_set))]
    test_mask = seed_mask[int(n_samples_first_snapshot * (1 - percentage_test_set)): n_samples_first_snapshot]
    snapshot_masks.append({"train": k_hop_subgraph(data, train_mask, k)[1], "test": k_hop_subgraph(data, test_mask, k)[1]})
    # remaining snapshots
    splitting_size = size - n_samples_first_snapshot
    split_size = int(splitting_size / n_snapshot)
    for i in range(n_snapshot):
        train_mask = seed_mask[int(i * split_size) + n_samples_first_snapshot: int(((i + 1) * split_size) - (split_size * (percentage_test_set)) + n_samples_first_snapshot)]
        test_mask = seed_mask[int(((i + 1) * split_size) - (split_size * (percentage_test_set)) + n_samples_first_snapshot): int((i + 1) * split_size) + n_samples_first_snapshot]
        snapshot_masks.append({"train": k_hop_subgraph(data, train_mask, k)[1], "test": k_hop_subgraph(data, test_mask, k)[1]})

    return data, target_type, snapshot_masks


def _fixed_randperm(n, k):
    fixed = torch.arange(k)
    suffix = torch.randperm(n - k) + k
    return torch.cat([fixed, suffix])