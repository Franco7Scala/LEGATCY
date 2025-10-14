import os

from torch_geometric.datasets import IMDB
from torch_geometric.datasets.dblp import DBLP
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
    for i in range(n_snapshot):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[int(i * size / n_snapshot): int((i + 1) * size / n_snapshot)] = 1
        loader = HGTLoader(
            data,
            num_samples={ntype: [99999] * 2 for ntype in dataset.data.node_types},  # full (non-sampled) neighborhood 2-hop
            input_nodes=(target_type, mask)  # core nodes per type
        )
        snapshots.append(next(iter(loader)))

    return snapshots


load_dataset("imdb", 2)