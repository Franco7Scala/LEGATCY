from sampling_strategies.abstract_strategy import AbstractStrategy


class DyHANE(AbstractStrategy):

    def __init__(self):
        super(DyHANE).__init__()

    def sample(self, n_split, data, new_nodes, new_edges, old_nodes, old_edges, target_type):
        pass