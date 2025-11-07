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
    