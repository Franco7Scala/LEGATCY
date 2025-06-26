import torch
import pandas as pd
import os

from src.data.data_utils import open_pickle, save_dict_to_pickle
from src.support.utils import get_base_dir



dataset_name = "openalex"
type_nodes = "old"

for i in range(7):
    name = os.path.join(get_base_dir(), f"openalex/snapshot_{i}/heterodata/", f"K_{type_nodes}_nodes.pkl")
    nodes = open_pickle(name)

    for key in nodes.keys():
        nodes[key[:len(key) - 1]] = nodes.pop(key)

    #save_dict_to_pickle(nodes, name)

















dataset_name = "openalex"
mapping_labels = open_pickle(os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))

for i in range(7):
    df = pd.read_csv(os.path.join(get_base_dir(), f"openalex/snapshot_{i}/original_data/", "author_labels.csv"))
    df['label'] = df['label'].map(mapping_labels)
    Y = torch.tensor(df['label'].values)
    torch.save(Y, os.path.join(get_base_dir(), f"openalex/snapshot_{i}/heterodata/author_labels.pt"))










dataset_name = "openalex"
mapping_labels = open_pickle(os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))
mapping_labels["generic"] = 22

save_dict_to_pickle(mapping_labels, os.path.join(get_base_dir(), dataset_name, 'mapping_labels.pkl'))

























path_to_swap_src = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/heterodata/edgelists/author_writes_paper.pt"
path_to_swap_dst = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/heterodata/edgelists/paper_is_written_by_author.pt"

reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)

reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)
reversed_edge_index = torch.load(path_to_swap_dst)


edge_index = torch.load(path_to_swap_src)

reversed_edge_index = torch.zeros(edge_index.shape, dtype=edge_index.dtype)
reversed_edge_index[0] = edge_index[1]
reversed_edge_index[1] = edge_index[0]

reversed_edge_index = torch.nan_to_num(reversed_edge_index)

torch.save(reversed_edge_index, path_to_swap_dst)

reversed_edge_index = torch.load(path_to_swap_dst)

print("Finito!")
