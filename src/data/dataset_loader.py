import os
import torch

from torch_geometric.datasets import IMDB
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.loader import HGTLoader

from src.data.data_utils import create_nodes_dict_empty
from src.support.utils import get_base_dir


def load_dataset(dataset_name, n_snapshot):
    path = os.path.join(get_base_dir(), dataset_name)
    snapshots = []

    if dataset_name == "imdb":
        dataset = IMDB(path)
        target_type = "movie"

    elif dataset_name == "dblp":
        dataset = DBLP(path)
        target_type = "paper"

    else:
        raise Exception(f"Dataset {dataset_name} not recognized or not implemented yet!")

    size = dataset.data[target_type].x.shape[0]
    if n_snapshot == 1:
         return [dataset.data]

    else:
        for i in range(n_snapshot):
            mask_tt = torch.arange(1, size) #torch.zeros(size, dtype=torch.bool)
            #mask_tt[int(i * size / n_snapshot): int((i + 1) * size / n_snapshot)] = 1
            mask_tt = mask_tt[int(i * size / n_snapshot): int((i + 1) * size / n_snapshot)]

            #mask = create_nodes_dict_empty(dataset.data)
            mask = {}
            mask[target_type] = mask_tt


            snapshots.append(dataset.data.subgraph(mask))

            '''loader = HGTLoader(
                dataset.data,
                num_samples = {ntype: [99999] * 2 for ntype in dataset.data.node_types},  # full (non-sampled) neighborhood 2-hop
                input_nodes = (target_type, mask)  # core nodes per type
            )
            snapshots.append(next(iter(loader)))
            '''

    return snapshots


x = load_dataset("dblp", 3)
print()