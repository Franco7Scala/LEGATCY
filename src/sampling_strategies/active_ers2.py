import torch

from src.sampling_strategies.basic_ers2 import BasicERS2


class ActiveERS2(BasicERS2):

    def __init__(self, al_technique, k):
        super(ActiveERS2).__init__()
        self.al_technique = al_technique
        self.k = k
        self.splits = []

    def _select_old_nodes(self, current_split, tot_split, data, new_nodes, old_nodes):
        if current_split == 0:
            self._calculate_splits(tot_split, data, old_nodes)
        
        return self.splits[current_split]

    def _calculate_splits(self, tot_split, data, old_nodes):
        scores_nodes_of_type = []
        # iterating over all types of nodes
        for i in range(len(data.node_stores)):
            # iterating over all nodes of type to calculate the score
            for j in range(data.node_stores[i]["x"].shape[0]):
                if j in old_nodes[data.node_types[i]]:
                    score = self.al_technique.get_score(data.node_stores[i]["x"][j])
                    scores_nodes_of_type.append((data.node_types[i], j, score))

        # sorting nodes keeping index and related score
        sorted_indices = sorted(range(len(scores_nodes_of_type)), key=lambda j: scores_nodes_of_type[j][2], reverse=True)
        scores_nodes_of_type = [scores_nodes_of_type[j] for j in sorted_indices]

        # generating splits
        selected_nodes = scores_nodes_of_type[:self.k]
        split_size = int(len(selected_nodes)/tot_split)
        splits = []
        current_split = self._initialize_split_dict(data)
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

        self.splits = splits

    def _initialize_split_dict(self, data):
        result = {}
        for node_type in data.node_types:
            result[node_type] = []

        return result
