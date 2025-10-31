import torch

from lshashing import LSHRandom
from src.sampling_strategies.abstract_strategy import AbstractStrategy
from src.support.utils_graph import k_hop_subgraph


class HashingStrategy(AbstractStrategy):

    def __init__(self, model):
        super(HashingStrategy, self).__init__()
        self.model = model

    def sample(self, n_split, data, new_nodes, old_nodes, target_type, kwargs=None):
        embeddings = {}
        for j in old_nodes[target_type]:
            node_subgraph = k_hop_subgraph(data, target_type, [j], 2).to(data[data.node_types[0]].x.device)
            embedding = self.model(node_subgraph.x_dict, node_subgraph.edge_index_dict, embeddings_only=True)
            embeddings[embedding] = j

        hashes = self._sample(embeddings)
        selected_nodes = self._sampling_based_on_clusters(hashes)
        return self.split_selected_nodes(selected_nodes, n_split)

    def _sample(self, embeddings):
        # calculating LSH hashes
        sample_data = torch.stack(list(embeddings.keys())).cpu().numpy()
        lshashing = LSHRandom(sample_data, hash_len=8, num_tables=2)
        clusters = lshashing.tables[0].hash_table
        # sampling nodes from clusters




    def _split_selected_nodes(self,  selected_nodes, n_split):
        pass
