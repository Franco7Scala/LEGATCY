"""
from: data > dataset_name > snapshot_i > original_data (nodes, edges)
to: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping.pkl, K_old_nodes.pkl, K_new_nodes.pkl,
k_old_edges.pkl, K_new_edges.pkl)

N.B.:
mapping is in the form --> node_type (e.g. authors) : {str_id : id}
K_nodes is in the form --> node_type (e.g. authors): list of ids
K_edges is in the form --> edge_type (e.g. author_writes_paper): list of pairs (lists) of ids
"""

import os
import pandas as pd
import torch

from src.data_utils import open_pickle, save_dict_to_pickle, extract_edge_info, attributes_encoding, edges_encoding


def extract_knowledge(dataset_name, no_snapshot):
    original_dir = os.path.join('data', dataset_name, 'snapshot_'+str(no_snapshot), 'original_data')
    heterodata_dir = os.path.join('data', dataset_name, 'snapshot_'+str(no_snapshot), 'heterodata')
    heterodata_prev_dir = os.path.join('data', dataset_name, f'snapshot_{(no_snapshot-1)}', 'heterodata')

    # processing nodes + mapping
    K_new_nodes = {}
    K_old_nodes = {}

    if no_snapshot == 0:
        mapping = {}
        for fname in os.listdir(os.path.join(original_dir, 'nodes')):
            n_type = fname[:-4]  # remove the last 4 characters (".csv")
            f = os.path.join(original_dir, 'nodes', fname)
            df = pd.read_csv(f)
            id_column = next(col for col in df.columns if "id" in col.lower())
            mapping[n_type] = {row[id_column]: idx for idx, row in df.iterrows()}
            K_new_nodes[n_type] = mapping[n_type].values()
            K_old_nodes[n_type] = {}

            X = attributes_encoding(df=df)
            torch.save(X, os.path.join(heterodata_dir, 'features', n_type+'.pt' ))

    else:
        mapping = open_pickle(os.path.join(heterodata_prev_dir, 'mapping.pkl'))
        for fname in os.listdir(os.path.join(original_dir, 'nodes')):
            n_type = fname[:-4]  # remove the last 4 characters (".csv")
            map_n_type = mapping[n_type]
            max_id = max(map_n_type.values(), default=-1)
            df = pd.read_csv(os.path.join(original_dir, 'nodes', fname))
            id_column = next(col for col in df.columns if "id" in col.lower())
            K_new_nodes[n_type] = []
            for id_str in df[id_column]:
                if id_str not in map_n_type:
                    max_id += 1
                    map_n_type[id_str] = max_id
                K_new_nodes[n_type].append(map_n_type[id_str]) # add to K_new both unseen and changed nodes
            K_old_nodes[n_type] = [id for id in mapping[n_type].values() if id not in K_new_nodes[n_type]] # add to K_old all - K_new

            X = attributes_encoding(df=df)
            X_prev = torch.load(os.path.join(heterodata_prev_dir, 'features', n_type+'.pt'))
            #X = torch.cat((X_prev, X), dim=0)
            for id_str, id in map_n_type.items():
                if id_str in df[id_column].values:
                    # NEW: Use row from X if id_str is in df
                    row_index = df[df[id_column] == id_str].index[0]
                    X[id] = X[row_index]
                else:
                    # OLD: Use row from X_prev if id_str was already in map_n_type and not in df
                    X[id] = X_prev[map_n_type[id_str]]
            torch.save(X, os.path.join(heterodata_dir, 'features', n_type + '.pt'))

    # saving mapping, K_new_nodes, K_old_nodes
    save_dict_to_pickle(mapping, os.path.join(heterodata_dir, 'mapping.pkl'))
    save_dict_to_pickle(K_new_nodes, os.path.join(heterodata_dir, 'K_new_nodes.pkl'))
    save_dict_to_pickle(K_old_nodes, os.path.join(heterodata_dir, 'K_old_nodes.pkl'))

    # processing edges
    K_new_edges = {}
    K_old_edges = {}

    for fname in os.listdir(os.path.join(original_dir, 'edges')):
        e_type = fname[:-4]  # remove the last 4 characters (".csv")
        edge_info = extract_edge_info(fname)
        n_type_src = edge_info[0]
        n_type_tgt = edge_info[1]
        key_src = n_type_src + 's'
        key_tgt = n_type_tgt + 's'
        f = os.path.join(original_dir, 'edges', fname)
        df = pd.read_csv(f)
        df['src'] = df['src'].map(mapping[key_src])
        df['tgt'] = df['tgt'].map(mapping[key_tgt])
        K_new_edges[e_type] = df[['src', 'tgt']].values.tolist()
        Xe = edges_encoding(df=df)

        if no_snapshot==0:
            K_old_edges[e_type] = []
        else:
            old_old_edges = open_pickle(os.path.join(heterodata_prev_dir, 'K_old_edges.pkl'))[e_type]
            old_new_edges = open_pickle(os.path.join(heterodata_prev_dir, 'K_new_edges.pkl'))[e_type]
            K_old_edges[e_type] = old_old_edges + old_new_edges
            Xe_prev = torch.load(os.path.join(heterodata_prev_dir, 'edgelists', e_type+'.pt'))
            Xe = torch.cat((Xe_prev, Xe), dim=0)

        torch.save(Xe, os.path.join(heterodata_dir, 'edgelists', e_type + '.pt'))

    # saving K_new_edges, K_old_edges
    save_dict_to_pickle(K_new_edges, os.path.join(heterodata_dir, 'K_new_edges.pkl'))
    save_dict_to_pickle(K_old_edges, os.path.join(heterodata_dir, 'K_old_edges.pkl'))