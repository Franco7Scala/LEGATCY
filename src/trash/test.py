import os
import torch
import pandas as pd

from data.data_utils import open_pickle
from utils import get_base_dir


mapp = open_pickle(os.path.join(get_base_dir(), "openalex", "snapshot_0", "heterodata", "mapping.pkl"))
edge_df = pd.read_csv(os.path.join(get_base_dir(), "openalex", "snapshot_0", "original_data", "edges", "author_writes_paper.csv"))
edge_tensor = torch.load(os.path.join(get_base_dir(), "openalex", "snapshot_0", "heterodata", "edgelists", "author_writes_paper.pt"))
print(edge_tensor.min())




