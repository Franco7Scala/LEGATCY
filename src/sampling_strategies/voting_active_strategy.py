import torch
import random

from sklearn.cluster import HDBSCAN
from src.support.utils_data import create_nodes_dict_empty
from src.sampling_strategies.basic_ers2 import BasicERS2
from src.support.utils_graph import k_hop_subgraph


class VotingActiveStrategy(VotingStrategy):

    def __init__(self, k=0, al_technique=None):
        super(VotingActiveStrategy, self).__init__(k)
        self.al_technique = al_technique

    def _sample_from_clusters(self, clusters):
        data = None
        selected_nodes = []
        n_tot_samples = sum(list(len(clusters[cluster]) for cluster in clusters.keys()))

        for cluster in clusters.keys():
            current_cluster_data = k_hop_subgraph(data, clusters[cluster], 2)[0].to(data.device)
            scores = self.al_technique.get_score(current_cluster_data, data.target_type)
            node_scores = []
            for idx, score in enumerate(scores):
                node_scores.append((current_cluster_data[data.target_type][idx].item(), score))

            n_to_sample = max(1, int(self.k * (len(clusters[cluster]) / n_tot_samples)))
            # sorting nodes keeping index and related score
            selected_nodes.extend(sorted(range(len(node_scores)), key=lambda j: node_scores[j][1], reverse=True)[:n_to_sample])

        return selected_nodes
