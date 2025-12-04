import torch
import os

from src.data.data_utils import create_nodes_dict_empty
from src.sampling_strategies.basic_ers2 import BasicERS2
from src.support.utils import predictable_hash
from src.support.utils_graph import k_hop_subgraph


class ActiveERS2(BasicERS2):

    def __init__(self, k=0, al_technique=None):
        super(ActiveERS2, self).__init__()
        self.al_technique = al_technique
        self.k = k
        self.splits = []

    def _select_old_nodes(self, current_split, n_split, data, new_nodes, old_nodes, kwargs=None):
        if current_split == 0:
            self._calculate_splits(n_split, data, old_nodes)

        if len(self.splits) <= current_split:
            return create_nodes_dict_empty(data, torch.tensor)

        return self.splits[current_split]

    def _calculate_splits(self, n_split, data, old_nodes):
        if len(old_nodes[data.target_type]) <= 0:
            self.splits = [create_nodes_dict_empty(data, torch.tensor)]

        else:
            node_scores = []
            node_subgraph = k_hop_subgraph(data, old_nodes[data.target_type], 2)[0].to(data.device)
            scores = self.al_technique.get_score(node_subgraph, data.target_type)
            for idx, score in enumerate(scores):
                node_scores.append((old_nodes[data.target_type][idx].item(), score))

            # sorting nodes keeping index and related score
            selected_nodes = sorted(range(len(node_scores)), key=lambda j: node_scores[j][1], reverse=True)[:self.k]
            # generating splits
            self.splits = self._split_selected_nodes(data, torch.tensor(selected_nodes).to(data.device), n_split)
