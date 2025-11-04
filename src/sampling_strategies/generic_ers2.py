import torch

from src.sampling_strategies.abstract_strategy import AbstractStrategy
from src.support.utils_graph import k_hop_subgraph


class GenericERS2(AbstractStrategy):

    def __init__(self):
        super(GenericERS2, self).__init__()

    def sample(self, n_split, data, new_nodes, old_nodes, target_type, kwargs=None):
        result = []
        self.subgraphs_dir = kwargs.subgraphs_dir
        # iterating over all the splits to generate
        for split in range(n_split):
            sampling_mask = {}
            # taking new nodes
            new_nodes_typed = self._select_new_nodes(split, n_split, data, new_nodes, old_nodes, target_type, kwargs)
            # taking old nodes
            old_nodes_typed = self._select_old_nodes(split, n_split, data, new_nodes, old_nodes, target_type, kwargs)
            # adding them to the mask
            for n_type in data.x_dict:
                sampling_mask[n_type] = torch.cat((new_nodes_typed[n_type].to(torch.int), old_nodes_typed[n_type].to(torch.int)))

            # applying sampling mask to the data generating a split ready for the training
            result.append(k_hop_subgraph(data, target_type, sampling_mask[target_type], 2))

        return result

    def _select_new_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type, kwargs):
        pass

    def _select_old_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type, kwargs):
        pass
