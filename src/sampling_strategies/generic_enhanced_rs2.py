import torch

from src.data_utils import extract_heterodata_sub
from src.sampling_strategies.abstract_strategy import AbstractStrategy


class GenericEnhancedRS2(AbstractStrategy):

    def sample(self, n_split, data, new_nodes, new_edges, old_nodes, old_edges):
        result = []
        # iterating over all the splits to generate
        for split in range(n_split):
            sampling_mask = {}
            # iterating over all the types of nodes
            for n_type in data.x_dict:
                current_nodes = torch.zeros(len(data.x_dict[n_type]), dtype=torch.bool)
                # taking new nodes
                current_nodes[new_nodes[n_type]] = True
                # taking old nodes
                self._select_old_nodes(current_nodes, n_type, split, n_split, data, new_nodes, new_edges, old_nodes, old_edges)
                # adding them to the mask
                sampling_mask[n_type] = current_nodes

            # applying sampling mask to the data generating a split ready for the training
            result.append(extract_heterodata_sub(data, sampling_mask))

        return result

    def _select_old_nodes(self, current_nodes, n_type, split, n_split, data, new_nodes, new_edges, old_nodes, old_edges):
        pass
