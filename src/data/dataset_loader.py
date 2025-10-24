import os
import torch

from torch_geometric.datasets import IMDB
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.loader import HGTLoader

from src.data.data_utils import create_nodes_dict_empty
from src.support.utils import get_base_dir
from src.support.utils_graph import k_hop_subgraph


def load_dataset(dataset_name, n_snapshot, k):
    path = os.path.join(get_base_dir(), dataset_name)
    snapshots = []

    if dataset_name == "imdb":
        dataset = IMDB(path)
        target_type = "movie"

    elif dataset_name == "dblp":
        dataset = DBLP(path)
        target_type = "paper"

    else:
        raise Exception(f"Unknown dataset '{dataset_name}'!")

    size = dataset.data[target_type].x.shape[0]
    if n_snapshot == 1:
         return [dataset.data]

    else:
        for i in range(n_snapshot):
            mask = torch.arange(1, size)
            mask = mask[int(i * size / n_snapshot): int((i + 1) * size / n_snapshot)]
            snapshots.append(k_hop_subgraph(dataset.data, target_type, mask, k))

    return dataset.data, snapshots


d, x = load_dataset("dblp", 3, 2) #TODO fare bene la divisione fra test e train
print()    #FIXME bug DBLP dataset conference node senza features lo fa scoppiare