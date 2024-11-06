"""
from: data > dataset_name > snapshot_i > original_data (nodes, edges)
to: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping.pkl, K_old.pkl, K_new.pkl)
"""

import os
import pandas as pd
import torch

from src.data_utils import open_pickle, save_dict_to_pickle, extract_edge_info, find_mapping, encoding_attributes


def extract_knowledge(dataset_name, no_snapshot):
    original_dir = os.path.join('data', dataset_name, 'snapshot_'+str(no_snapshot), 'original_data')
    heterodata_dir = os.path.join('data', dataset_name, 'snapshot_'+str(no_snapshot), 'heterodata')
    heterodata_prev_dir = os.path.join('data', dataset_name, f'snapshot_{(no_snapshot-1)}', 'heterodata')

    #mapping
    if no_snapshot == 0:
        mapping = {}
        K_new = {}
        K_old = {}
        for fname in os.listdir(os.path.join(original_dir, 'nodes')):
            n_type = fname[:-4]  # remove the last 4 characters (".csv")
            f = os.path.join(original_dir, 'nodes', fname)
            df = pd.read_csv(f)
            id_column = next(col for col in df.columns if "id" in col.lower())
            mapping[n_type] = {row[id_column]: idx for idx, row in df.iterrows()}
            K_new[n_type] = mapping[n_type]
            K_old[n_type] = {}

            X = encoding_attributes(df=df)
            torch.save(X, os.path.join(heterodata_dir, 'features', n_type+'.pt' ))

        save_dict_to_pickle(mapping, os.path.join(heterodata_dir, 'mappings.pkl'))
        save_dict_to_pickle(K_new, os.path.join(heterodata_dir, 'K_new.pkl'))
        save_dict_to_pickle(K_old, os.path.join(heterodata_dir, 'K_old.pkl'))

    else:
        mapping = open_pickle(os.path.join(heterodata_prev_dir, 'mapping.pkl'))
        for fname in os.listdir(os.path.join(original_dir, 'nodes')):
            n_type = fname[:-4]  # remove the last 4 characters (".csv")
            map_n_type = mapping[n_type]
            # da qui va visto
            df = pd.read_csv(os.path.join(original_dir, 'nodes', fname))
            find_mapping(id, mapping)

        # il mapping inizia da previous

        # K_new = TUTTI
        # K_old = vuota

    for fname in os.listdir(os.path.join(original_dir, 'edges')):
        edge_info = extract_edge_info(fname)
        n_type_src = edge_info[0]
        n_type_tgt = edge_info[1]
        key_src = n_type_src + 's'
        key_tgt = n_type_tgt + 's'
        f = os.path.join(original_dir, 'edges', fname)
        df = pd.read_csv(f)
        df['src'] = df['src'].map(mapping[key_src])
        df['tgt'] = df['tgt'].map(mapping[key_tgt])
