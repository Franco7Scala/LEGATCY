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

    in_dim = 128
    embeddings = nn.ModuleDict()
    for ntype in dataset.data.metadata()[0]:  # metadata()[0] returns node types list
        if 'x' not in dataset.data[ntype]:

            num_nodes = dataset.data[ntype].num_nodes
            dataset.data[ntype].x = torch.zeros((num_nodes, in_dim), device=device)
            #dataset.data[ntype].x = nn.Embedding(num_nodes, in_dim)(torch.arange(num_nodes)).detach()


    metapaths = get_metapaths(dataset_name)
    dataset.data.mps = metapaths
    dataset.data = AddMetaPaths(metapaths=metapaths, weighted=True)(dataset.data)
    dataset.data.target_type = target_type
    dataset.data.to(device)
    dataset.data.device = dataset.data.x_dict[dataset.data.target_type].device

    size = dataset.data[target_type].x.shape[0]
    for i in range(n_snapshot):
        mask = torch.arange(0, size)
        mask = mask[int(i * size / n_snapshot): int((i + 1) * size / n_snapshot)]
        snapshot_masks.append(k_hop_subgraph(dataset.data, mask, k)[1])

    return dataset.data, target_type, snapshot_masks
