from torch_geometric.data import HeteroData
import torch
import torch_geometric.explain
from torch_geometric.nn import to_hetero

from data.data_utils import create_nodes_dict_empty
from sampling_strategies.abstract_strategy import AbstractStrategy
from sampling_strategies.basic_ers2 import BasicERS2
from support.utils import get_metapaths, get_device
from support.utils_graph import extract_edges

from src.support.utils_graph import k_hop_subgraph


class DyHANE(BasicERS2):

    def __init__(self, k=0):
        super(DyHANE, self).__init__()
        self.k = k

    def sample(self, n_split, data, new_nodes, old_nodes, kwargs=None):
        target_type = data.target_type
        result = []
        # iterating over all the splits to generate
        for split in range(n_split):
            sampling_mask = {}
            # taking new nodes (new+changed nodes --- influenced nodes)
            metapaths = data.mps

            new_edges = extract_edges(data, new_nodes) # these are the edges incident to at least one new node

            new_nodes_typed = self._select_new_nodes(1, n_split, data, new_nodes, old_nodes, kwargs)
            # taking old nodes
            if hasattr(kwargs, "old_model"):
                old_data = k_hop_subgraph(data, old_nodes[target_type], 2)[0].to(data.device)
                old_nodes_typed = self._select_old_nodes(kwargs.old_model, old_data)
            else:
                old_nodes_typed = self._initialize_split_dict(data, dtype=torch.tensor)
            # adding them to the mask
            for n_type in data.x_dict:
                sampling_mask[n_type] = torch.cat(
                    (new_nodes_typed[n_type].to(torch.int), old_nodes_typed[n_type].to(torch.int)))

            # applying sampling mask to the data generating a split ready for the training
            result.append((data.subgraph(sampling_mask), sampling_mask))

        return result

    # helpe -- get or create set
    def _safe_add(self, node_dict, ntype):
        if ntype not in node_dict:
            node_dict[ntype] = set()

        return node_dict[ntype]

    """
    Args: data (HeteroData object), new_edges (dict { (src_type, rel_type, dst_type): [(u,v), ...] }),
    metapaths (list of metapaths, each is a list of edge types, 
    e.g. [(target_type,'writes','paper'), ('paper','is_written_by',target_type)])

    Returns: dict { node_type: [node_ids] }
    """
    # TODO NOTA: sono i "nuovi" nuovi nodi --- cioè i nodi completamente nuovi e quelli aggiornati

    def _select_new_nodes_che_non_serve(self, data: HeteroData, new_edges, metapaths):

        node_dict = {}
        target_type = data.target_type

        # Precompute adjacency per edge type for quick lookup
        neighbors = {}
        for rel in data.edge_types:
            srcs, dsts = data[rel].edge_index
            neighbors[rel] = {}
            for s, d in zip(srcs.tolist(), dsts.tolist()):
                if s not in neighbors[rel]:
                    neighbors[rel][s] = []
                neighbors[rel][s].append(d)

        # Precompute metapath-based adjacency
        metapath_edges = {}
        for meta_id in range(len(metapaths)):
            meta_key = (target_type, f'metapath_{meta_id}', target_type)
            if meta_key in data:
                srcs, dsts = data[meta_key].edge_index
                metapath_edges[meta_key] = set(map(tuple, zip(srcs.tolist(), dsts.tolist())))

        # For each new edge
        for etype, edges in new_edges.items():
            src_type, _, dst_type = etype
            for (u, v) in edges:
                self._safe_add(node_dict, src_type).add(u)
                self._safe_add(node_dict, dst_type).add(v)

                creates_new = False

                # Check all metapaths
                for meta_id, metapath in enumerate(metapaths):
                    meta_key = (target_type, f'metapath_{meta_id}', target_type)
                    if meta_key not in metapath_edges:
                        continue

                    for pos, rel in enumerate(metapath):
                        if rel != etype:
                            continue

                        left_nodes = [u]
                        right_nodes = [v]

                        # Walk backwards
                        for r in reversed(metapath[:pos]):
                            new_left = []
                            for node in left_nodes:
                                if node in neighbors.get(r, {}):
                                    new_left.extend(neighbors[r][node])
                            left_nodes = new_left

                        # Walk forwards
                        for r in metapath[pos + 1:]:
                            new_right = []
                            for node in right_nodes:
                                if node in neighbors.get(r, {}):
                                    new_right.extend(neighbors[r][node])
                            right_nodes = new_right

                        if metapath[0][0] == target_type and metapath[-1][2] == target_type:
                            for a1 in left_nodes:
                                for a2 in right_nodes:
                                    if (a1, a2) not in metapath_edges[meta_key]:
                                        creates_new = True

                # If new meta-paths appear, expand neighborhood
                if creates_new:
                    # one-hop neighbors
                    for rel in data.edge_types:
                        srcs, dsts = data[rel].edge_index
                        if rel[0] == src_type:
                            self._safe_add(node_dict, rel[2]).update(dsts[srcs == u].tolist())
                        if rel[2] == src_type:
                            self._safe_add(node_dict, rel[0]).update(srcs[dsts == u].tolist())
                        if rel[0] == dst_type:
                            self._safe_add(node_dict, rel[2]).update(dsts[srcs == v].tolist())
                        if rel[2] == dst_type:
                            self._safe_add(node_dict, rel[0]).update(srcs[dsts == v].tolist())

                    # meta-path neighbors for target type
                    for node, ntype in [(u, src_type), (v, dst_type)]:
                        if ntype == target_type:
                            for meta_key in metapath_edges.keys():
                                srcs, dsts = data[meta_key].edge_index
                                self._safe_add(node_dict, target_type).update(dsts[srcs == node].tolist())
                                self._safe_add(node_dict, target_type).update(srcs[dsts == node].tolist())

        return {ntype: torch.tensor(list(ids)).to(data.device) for ntype, ids in node_dict.items()}


    def _get_explainer(self, model, data, target_type):
        explainer = torch_geometric.explain.Explainer(
            model=model,
            algorithm=torch_geometric.explain.CaptumExplainer('IntegratedGradients'),  # InputXGradient
            explanation_type='phenomenon',  # model's beahviour (model) vs individual predictions (phenomenon)
            node_mask_type='attributes',
            edge_mask_type=None,
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='probs',  # log_probs, raw
            ),
        )
        num_target_nodes = torch.arange(data.x_dict[target_type].shape[0])
        with torch.no_grad():
            explainer = explainer(data.x_dict, data.edge_index_dict, target=data.y_dict[target_type]) #torch_geometric.explain.Explainer
        return explainer

    #nota: sono i nodi più "significativi" del vecchio modello
    def _select_old_nodes(self, old_model, old_data): # buffer_size=768, topk=64

        old_nodes = self._initialize_split_dict(old_data, dtype=torch.tensor)

        target_type = old_data.target_type
        explanation = self._get_explainer(old_model, old_data, target_type)
        values, indices = torch.topk(explanation.node_mask_dict[target_type].sum(-1), k=self.k)
        old_nodes[target_type] = indices  # .cpu().numpy().tolist()

        """
        n_types = list(old_data.x_dict.keys())
        percs = [0] * len(n_types)
        explanation = self._get_explainer(old_model, old_data, target_type)

        # type importance, dato dalla media dei topk per ogni tipo
        for i, nt in enumerate(n_types):
            values, indices = torch.topk(explanation.node_mask_dict[nt].sum(-1), k=topk)
            percs[i] = values.mean().item()

        scaling_factor = buffer_size / sum(percs)
        rates = [round(val * scaling_factor) for val in percs]

        # top nodes for each type
        for i, nt in enumerate(n_types):
            values, indices = torch.topk(explanation.node_mask_dict[nt].sum(-1), k=rates[i])
            old_nodes[nt] = indices #.cpu().numpy().tolist()
        """

        return old_nodes




