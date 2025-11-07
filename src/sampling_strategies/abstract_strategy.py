import torch


class AbstractStrategy:

    def __init__(self):
        self.needs_previous_model = True

    def sample(self, n_split, data, new_nodes, old_nodes, kwargs=None):
        pass

    def _initialize_split_dict(self, data, dtype=list):
        result = {}
        for node_type in data.node_types:
            if dtype == list:
                result[node_type] = []

            elif dtype == torch.tensor:
                result[node_type] = torch.tensor([], dtype=torch.int).to(data.device)

            else:
                raise ValueError("Type not allowed!")

        return result

    def _split_selected_nodes(self, data, selected_nodes, n_split):
        result = []
        split_size = int(len(selected_nodes) / n_split)
        for split in range(n_split):
            sampling_mask = self._initialize_split_dict(data, torch.tensor)
            sampling_mask[data.target_type] = selected_nodes[split_size * split: split_size * (split + 1)]
            result.append(sampling_mask)

        return result
