import torch
import os

from src.sampling_strategies.basic_ers2 import BasicERS2
from src.support.utils import predictable_hash
from src.support.utils_graph import k_hop_subgraph


class ActiveERS2(BasicERS2):

    def __init__(self, k=0, al_technique=None):
        super(ActiveERS2, self).__init__()
        self.al_technique = al_technique
        self.k = k
        self.splits = []

    def _select_old_nodes(self, current_split, tot_split, data, new_nodes, old_nodes, target_type):
        if current_split == 0:
            self._calculate_splits(tot_split, data, old_nodes, target_type)

        if len(self.splits) <= current_split:
            return self._initialize_split_dict(data, torch.tensor)

        return self.splits[current_split]

    def _calculate_splits(self, tot_split, data, old_nodes, target_type):
        scores_nodes_of_type = []
        # iterating over all types of nodes
        for i in range(len(data.node_stores)):
            # iterating over all nodes of type to calculate the score
            for j in range(data.node_stores[i]["x"].shape[0]):
                if data.node_types[i] == target_type and not data[target_type].train_mask[j]:
                    continue

                if j in old_nodes[data.node_types[i]]:
                    subset_dict = {}
                    for node_type in data.node_types:
                        if node_type == data.node_types[i]:
                            subset_dict[node_type] = torch.tensor([j]).to(torch.int).to(data[data.node_types[0]].x.device)

                        else:
                            subset_dict[node_type] = torch.tensor([]).to(torch.int).to(data[data.node_types[0]].x.device)

                    subgraph = self._extract_subgraph(data, target_type, subset_dict)
                    score = self.al_technique.get_score(subgraph, target_type)
                    scores_nodes_of_type.append((data.node_types[i], j, score))

        # sorting nodes keeping index and related score
        sorted_indices = sorted(range(len(scores_nodes_of_type)), key=lambda j: scores_nodes_of_type[j][2], reverse=False)
        scores_nodes_of_type = [scores_nodes_of_type[j] for j in sorted_indices]
        # generating splits
        selected_nodes = scores_nodes_of_type[:self.k]
        split_size = int(len(selected_nodes)/tot_split)
        splits = []
        current_split = self._initialize_split_dict(data, list)
        # iterating over all the selected nodes
        for i in range(len(selected_nodes)):
            current_type = selected_nodes[i][0]
            current_index = selected_nodes[i][1]
            current_split[current_type].append(current_index)
            # checking if split reach the target size
            if sum([len(current_split[val]) for val in current_split.keys()]) >= split_size:
                # converting lists to tensors
                tensored_split = {}
                for key in current_split.keys():
                    tensored_split[key] = torch.tensor(current_split[key]).to(data[data.node_types[0]].x.device).to(torch.int)

                # adding split to the splits set
                splits.append(tensored_split)
                current_split = self._initialize_split_dict(data)

        if len(splits) > 0:
            self.splits = splits

        else:
            self.splits = [self._initialize_split_dict(data, torch.tensor)]

    def _extract_subgraph(self, data, target_type, subset_dict):
        path_subgraph = f"{self.subgraphs_dir}/{predictable_hash(str(subset_dict))}.sg"
        if os.path.exists(path_subgraph):
            subgraph = torch.load(path_subgraph, weights_only=False)

        else:
            subgraph = k_hop_subgraph(data, target_type, subset_dict[target_type], 2).to(data[data.node_types[0]].x.device)
            torch.save(subgraph, path_subgraph)

        return subgraph
