import torch

from src.sampling_strategies.generic_ers2 import GenericERS2


class FullRetraining(GenericERS2):

    def __init__(self):
        super(GenericERS2, self).__init__()
        self.needs_previous_model = False

    def _select_new_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type, kwargs=None):
        return self._select_nodes(data, new_nodes)

    def _select_old_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type, kwargs=None):
        return self._select_nodes(data, old_nodes)

    def _select_nodes(self, data, nodes):
        result = {}
        for n_type in data.x_dict:
            if len(nodes[n_type]) == 0:
                result[n_type] = torch.tensor([], dtype=torch.int).to(data[data.node_types[0]].x.device)

            else:
                result[n_type] = torch.tensor(nodes[n_type]).to(data[data.node_types[0]].x.device).to(torch.int)

        return result
