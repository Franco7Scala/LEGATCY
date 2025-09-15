from src.data.data_utils import create_nodes_dict_empty
from src.sampling_strategies.other_full_retraining import FullRetraining


class OnlineTraining(FullRetraining):

    def __init__(self):
        super(OnlineTraining, self).__init__()

    def _select_old_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type):
        return self._select_nodes(data, create_nodes_dict_empty(data))
