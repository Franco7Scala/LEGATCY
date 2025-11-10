import os
import torch

from torch_geometric.datasets import IMDB
from torch_geometric.datasets.dblp import DBLP
from src.data.data_utils import create_nodes_dict_empty
from src.support.utils import get_base_dir
from src.support.utils_graph import k_hop_subgraph
from support.utils import get_metapaths


def load_dataset(dataset_name, n_snapshot, k, device):
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

    dataset.data.mps = get_metapaths(dataset_name)
    dataset.data.target_type = target_type
    dataset.data.to(device)
    dataset.data.device = dataset.data.x_dict[dataset.data.target_type].device

    size = dataset.data[target_type].x.shape[0]
    for i in range(n_snapshot):
        mask = torch.arange(0, size)
        mask = mask[int(i * size / n_snapshot): int((i + 1) * size / n_snapshot)]
        snapshot_masks.append(k_hop_subgraph(dataset.data, mask, k)[1])

    return dataset.data, target_type, snapshot_masks
