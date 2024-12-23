import torch
import pickle

from src.data_utils import get_base_dir
from src.sampling_strategies.abstract_strategy import AbstractStrategy


class GenericERS2(AbstractStrategy):

    def __init__(self):
        super(GenericERS2).__init__()

    def sample(self, n_split, data, new_nodes, new_edges, old_nodes, old_edges, target_type, save=False, cycle=0):
        result = []
        # iterating over all the splits to generate
        for split in range(n_split):
            sampling_mask = {}
            # taking new nodes
            new_nodes_typed = self._select_new_nodes(split, n_split, data, new_nodes, old_nodes)
            # taking old nodes
            old_nodes_typed = self._select_old_nodes(split, n_split, data, new_nodes, old_nodes, target_type)
            # adding them to the mask
            for n_type in data.x_dict:
                sampling_mask[n_type] = torch.cat((new_nodes_typed[n_type], old_nodes_typed[n_type]))

            # applying sampling mask to the data generating a split ready for the training
            if save:
                self._save_sub_graph(sampling_mask, n_split, cycle)

            result.append(data.subgraph(sampling_mask))

        return result

    def _save_sub_graph(self, dict_selected, n_split, cycle):
        res = {}
        for key in dict_selected.keys():
            res[key] = dict_selected[key].tolist()

        with open(f"{get_base_dir()}/mumin/selected_{n_split}_{cycle}.pkl", "wb") as handle: #TODO fa schifo questo codice come la conferenza dei sociologi che non capiscono un cazzo
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _select_new_nodes(self, split, n_split, data, new_nodes, old_nodes):
        pass

    def _select_old_nodes(self, split, n_split, data, new_nodes, old_nodes, target_type):
        pass
