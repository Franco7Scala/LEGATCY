import torch

from src.sampling_strategies.generic_ers2 import GenericERS2


class BasicERS2(GenericERS2):

    def __init__(self):
        super(BasicERS2).__init__()

    def _select_new_nodes(self, current_split, tot_split, data, new_nodes, old_nodes):
        return torch.tensor(new_nodes).to(data[data.node_types[0]].x.device).to(torch.int)

    def _select_old_nodes(self, current_split, tot_split, data, new_nodes, old_nodes):
        len_split = max(int(len(old_nodes) / tot_split), len(old_nodes))
        return torch.tensor(old_nodes[(len_split * current_split): ((len_split * current_split) + len_split)]).to(data[data.node_types[0]].x.device).to(torch.int)
