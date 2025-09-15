from torch_geometric.data import HeteroData
import torch
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.nn import to_hetero

from sampling_strategies.abstract_strategy import AbstractStrategy
from support.utils import get_metapaths


class DyHANE(AbstractStrategy):

    def __init__(self):
        super(DyHANE).__init__()

    def sample(self, n_split, data, new_nodes, new_edges, old_nodes, old_edges, target_type):
        result = []
        # iterating over all the splits to generate
        for split in range(n_split):
            sampling_mask = {}
            # taking new nodes (new+changed nodes --- influenced nodes)
            metapaths = get_metapaths("openalex") #TODO
            new_nodes_typed = self._select_new_nodes(data, new_edges, metapaths)
            # taking old nodes
            old_nodes_typed = self._select_old_nodes(split, n_split, data, new_nodes, old_nodes, target_type)
            # adding them to the mask
            for n_type in data.x_dict:
                sampling_mask[n_type] = torch.cat(
                    (new_nodes_typed[n_type].to(torch.int), old_nodes_typed[n_type].to(torch.int)))

            # applying sampling mask to the data generating a split ready for the training
            result.append(data.subgraph(sampling_mask))

        return result

    # helpe -- get or create set
    def _safe_add(self, node_dict, ntype):
        if ntype not in node_dict:
            node_dict[ntype] = set()

        return node_dict[ntype]

    """
    Args: data (HeteroData object), new_edges (dict { (src_type, rel_type, dst_type): [(u,v), ...] }),
    metapaths (list of metapaths, each is a list of edge types, 
    e.g. [('author','writes','paper'), ('paper','is_written_by','author')])

    Returns: dict { node_type: [node_ids] }
    """
    # TODO NOTA: sono i "nuovi" nuovi nodi --- cioè i nodi completamente nuovi e quelli aggiornati
    def _select_new_nodes(self, data: HeteroData, new_edges, metapaths):

        node_dict = {}

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
            meta_key = ('author', f'metapath_{meta_id}', 'author')
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
                    meta_key = ('author', f'metapath_{meta_id}', 'author')
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

                        if metapath[0][0] == 'author' and metapath[-1][2] == 'author':
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

                    # meta-path neighbors for authors
                    for node, ntype in [(u, src_type), (v, dst_type)]:
                        if ntype == 'author':
                            for meta_key in metapath_edges.keys():
                                srcs, dsts = data[meta_key].edge_index
                                self._safe_add(node_dict, 'author').update(dsts[srcs == node].tolist())
                                self._safe_add(node_dict, 'author').update(srcs[dsts == node].tolist())

        return {ntype: sorted(list(ids)) for ntype, ids in node_dict.items()}


        def _get_explainer(model, data, target_type):
            explainer = Explainer(
                model=model,
                algorithm=CaptumExplainer('IntegratedGradients'),  # InputXGradient
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
                explainer = explainer(data.x_dict, data.edge_index_dict, target=data.y_dict[target_type], index=num_target_nodes)
            return explainer

        #TODO
        def _select_old_nodes(old_model, old_data, target_type, B_size=768, topk=64):

            old_nodes = {}

            n_types = list(data.x_dict.keys())
            percs = [0] * len(n_types)
            explanation = _get_explainer(old_model, old_data, target_type)

            # type importance, dato dalla media dei topk per ogni tipo
            for i, nt in enumerate(n_types):
                values, indices = torch.topk(explanation.node_mask_dict[nt].sum(-1), k=topk)
                percs[i] = values.mean().item()

            scaling_factor = B_size / sum(percs)
            rates = [round(val * scaling_factor) for val in percs]

            # top nodes for each type
            for i, nt in enumerate(n_types):
                values, indices = torch.topk(explanation.node_mask_dict[nt].sum(-1), k=rates[i])
                old_nodes[nt] = indices.cpu().numpy().tolist()

            return old_nodes




