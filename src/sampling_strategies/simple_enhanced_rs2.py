import torch

from src.data_utils import extract_heterodata_sub
from src.sampling_strategies.abstract_strategy import AbstractStrategy


class SimpleEnhancedRS2(AbstractStrategy):

    def _select_old_nodes(self, current_nodes, n_type, split, n_split, data, new_nodes, new_edges, old_nodes, old_edges):
        len_split = int(len(data)/n_split)
        to_keep = old_nodes[n_type][(len_split * split): ((len_split * split) + len_split)]
        current_nodes[to_keep] = True
