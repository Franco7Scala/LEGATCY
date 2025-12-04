import torch

from src.data.data_utils import create_nodes_dict_empty


class AbstractStrategy:

    def __init__(self):
        self.needs_previous_model = True

    def sample(self, n_split, data, new_nodes, old_nodes, kwargs=None):
        pass

    def _split_selected_nodes(self, data, selected_nodes, n_split):
        result = []
        split_size = int(len(selected_nodes) / n_split)
        for split in range(n_split):
            sampling_mask = create_nodes_dict_empty(data, torch.tensor)
            sampling_mask[data.target_type] = selected_nodes[split_size * split: split_size * (split + 1)]
            result.append(sampling_mask)

        return result
