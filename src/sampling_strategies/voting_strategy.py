import torch
import random

from sklearn.cluster import HDBSCAN
from src.sampling_strategies.basic_ers2 import BasicERS2
from src.support.utils import plot
from src.support.utils_graph import k_hop_subgraph


class VotingStrategy(BasicERS2):

    def __init__(self, k=0):
        super(VotingStrategy, self).__init__()
        self.k = k

    def _select_old_nodes(self, current_split, n_split, data, new_nodes, old_nodes, target_type, kwargs=None):
        if current_split == 0:
            self._calculate_splits(n_split, data, old_nodes, target_type, kwargs)

        if len(self.splits) <= current_split:
            return self._initialize_split_dict(data, torch.tensor)

        return self.splits[current_split]

    def _calculate_splits(self, n_split, data, old_nodes, target_type, kwargs):
        embeddings = {}
        for j in old_nodes[target_type]:
            node_subgraph = k_hop_subgraph(data, target_type, torch.tensor([j]), 2)[0].to(data[data.node_types[0]].x.device)
            embedding = kwargs.old_model(node_subgraph.x_dict, node_subgraph.edge_index_dict, embeddings_only=True)
            embeddings[embedding[target_type][0]] = j.item()

        if len(embeddings) == 0:
            self.splits = [self._initialize_split_dict(data, torch.tensor)]

        else:
            nodes = self._clusterize_embeddings(embeddings)
            selected_nodes = self._sample_from_clusters(nodes)
            self.splits = self._split_selected_nodes(data, torch.tensor(selected_nodes).to(data[data.node_types[0]].x.device), target_type, n_split)

    def _clusterize_embeddings(self, embeddings):
        x = torch.stack(list(embeddings.keys())).detach().cpu().numpy()
        hdb = HDBSCAN().fit(x)

        indexes_for_clusters = {}
        for idx, label in enumerate(hdb.labels_):
            if label not in indexes_for_clusters:
                indexes_for_clusters[label] = []

            indexes_for_clusters[label].append(idx)

        return indexes_for_clusters

    def _sample_from_clusters(self, clusters):
        selected_nodes = []
        n_tot_samples = sum(list(len(clusters[cluster]) for cluster in clusters.keys()))

        for cluster in clusters.keys():
            random.shuffle(clusters[cluster])
            n_to_sample = max(1, int(self.k * (len(clusters[cluster]) / n_tot_samples)))
            selected_nodes.extend(clusters[cluster][:n_to_sample])

        return selected_nodes

    def _split_selected_nodes(self, data, supplementary_nodes, target_type, n_split):
        result = []
        split_size = int(len(supplementary_nodes) / n_split)
        for split in range(n_split):
            sampling_mask = self._initialize_split_dict(data, torch.tensor)
            sampling_mask[target_type] = supplementary_nodes[split_size * split : split_size * (split+1)]
            result.append(sampling_mask)

        return result
