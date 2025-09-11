from torch_geometric.data import HeteroData

from sampling_strategies.abstract_strategy import AbstractStrategy


class DyHANE(AbstractStrategy):

    def __init__(self):
        super(DyHANE).__init__()

    def sample(self, n_split, data, new_nodes, new_edges, old_nodes, old_edges, target_type):
        pass


    def check_new_metapath_instances(data: HeteroData, new_edges, metapaths):
        existing_metapath_edges = {}
        for key in data.metapath_dict.keys():
            edge_idx = data[key].edge_index
        existing_metapath_edges[key] = set(map(tuple, edge_idx.t().tolist()))

        results = {}
        for etype, edges in new_edges.items():
            results[etype] = []
            for u, v in edges:
                is_new = False

            for meta_id, metapath in enumerate(metapaths):
                meta_key = ('author', f'metapath_{meta_id}', 'author')
                if meta_key not in existing_metapath_edges:
                    continue

                if metapath == [('author', 'writes', 'paper'), ('paper', 'is_written_by', 'author')]:
                    if etype == ('paper', 'is_written_by', 'author'):
                        srcs, dsts = data[('author', 'writes', 'paper')].edge_index
                        mask = dsts == u
                        for a1 in srcs[mask].tolist():
                            if (a1, v) not in existing_metapath_edges[meta_key]:
                                is_new = True
                                break
                    elif etype == ('author', 'writes', 'paper'):
                        srcs, dsts = data[('paper', 'is_written_by', 'author')].edge_index
                        mask = srcs == v
                        for a2 in dsts[mask].tolist():
                            if (u, a2) not in existing_metapath_edges[meta_key]:
                                is_new = True
                                break

                elif metapath == [('author', 'is_affiliated_with', 'institution'),
                                  ('institution', 'is_affiliation_of', 'author')]:
                    if etype == ('author', 'is_affiliated_with', 'institution'):
                        srcs, dsts = data[('institution', 'is_affiliation_of', 'author')].edge_index
                        mask = srcs == v
                        for a2 in dsts[mask].tolist():
                            if (u, a2) not in existing_metapath_edges[meta_key]:
                                is_new = True
                                break
                    elif etype == ('institution', 'is_affiliation_of', 'author'):
                        srcs, dsts = data[('author', 'is_affiliated_with', 'institution')].edge_index
                        mask = dsts == u
                        for a1 in srcs[mask].tolist():
                            if (a1, v) not in existing_metapath_edges[meta_key]:
                                is_new = True
                                break

                elif metapath == [('author', 'writes', 'paper'), ('paper', 'cites', 'paper'),
                                  ('paper', 'is_written_by', 'author')]:
                    if etype == ('paper', 'cites', 'paper'):
                        a1s = data[('author', 'writes', 'paper')].edge_index[0][
                            data[('author', 'writes', 'paper')].edge_index[1] == u]
                        a2s = data[('paper', 'is_written_by', 'author')].edge_index[1][
                            data[('paper', 'is_written_by', 'author')].edge_index[0] == v]
                        for a1 in a1s.tolist():
                            for a2 in a2s.tolist():
                                if (a1, a2) not in existing_metapath_edges[meta_key]:
                                    is_new = True
                                    break

            results[etype].append(is_new)
        return results


    def collect_nodes(data: HeteroData, new_edges, metapaths):
        meta_results = check_new_metapath_instances(data, new_edges, metapaths)

        node_dict = {}

        for etype, edges in new_edges.items():
            src_type, _, dst_type = etype
            for idx, (u, v) in enumerate(edges):
                if src_type not in node_dict:
                    node_dict[src_type] = set()
                if dst_type not in node_dict:
                    node_dict[dst_type] = set()

                node_dict[src_type].add(u)
                node_dict[dst_type].add(v)

                if meta_results[etype][idx]:
                    # Add one-hop neighbors
                    for rel in data.edge_types:
                        srcs, dsts = data[rel].edge_index
                        if rel[0] == src_type:
                            if rel[2] not in node_dict:
                                node_dict[rel[2]] = set()
                            node_dict[rel[2]].update(dsts[srcs == u].tolist())
                        if rel[2] == src_type:
                            if rel[0] not in node_dict:
                                node_dict[rel[0]] = set()
                            node_dict[rel[0]].update(srcs[dsts == u].tolist())
                        if rel[0] == dst_type:
                            if rel[2] not in node_dict:
                                node_dict[rel[2]] = set()
                            node_dict[rel[2]].update(dsts[srcs == v].tolist())
                        if rel[2] == dst_type:
                            if rel[0] not in node_dict:
                                node_dict[rel[0]] = set()
                            node_dict[rel[0]].update(srcs[dsts == v].tolist())

                    # Add meta-path neighbors for authors
                    for node, ntype in [(u, src_type), (v, dst_type)]:
                        if ntype == 'author':
                            if 'author' not in node_dict:
                                node_dict['author'] = set()
                            for meta_id in range(len(metapaths)):
                                meta_key = ('author', f'metapath_{meta_id}', 'author')
                                if meta_key in data:
                                    srcs, dsts = data[meta_key].edge_index
                                    node_dict['author'].update(dsts[srcs == node].tolist())
                                    node_dict['author'].update(srcs[dsts == node].tolist())

        return {ntype: sorted(list(nodes)) for ntype, nodes in node_dict.items()}


