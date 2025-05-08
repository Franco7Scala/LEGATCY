import torch

from src.sampling_strategies.abstract_strategy import AbstractStrategy


class GenericERS2(AbstractStrategy):

    def __init__(self):
        super(GenericERS2).__init__()

    def sample(self, n_split, data, new_nodes, new_edges, old_nodes, old_edges, target_type):
        result = []
        # iterating over all the splits to generate
        for split in range(n_split):
            sampling_mask = {}
            # taking new nodes
            new_nodes_typed = self._select_new_nodes(split, n_split, data, new_nodes, old_nodes, target_type)
            # taking old nodes
            old_nodes_typed = self._select_old_nodes(split, n_split, data, new_nodes, old_nodes, target_type)
            # adding them to the mask
            for n_type in data.x_dict:
                sampling_mask[n_type] = torch.cat((new_nodes_typed[n_type].to(torch.int), old_nodes_typed[n_type].to(torch.int)))

            # applying sampling mask to the data generating a split ready for the training
            result.append(data.subgraph(sampling_mask))

        return result

    def _select_new_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type):
        pass

    def _select_old_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type):
        pass
