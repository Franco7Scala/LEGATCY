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

from src.data_utils import open_pickle, save_dict_to_pickle, get_target_type, extract_edge_info, attributes_encoding, edges_encoding
#from src.utils import Color, cprint


def extract_knowledge(dataset_name, no_snapshot):

    print(f"Processing snapshot {no_snapshot}")
    base_dir = '/mnt/nas/martirano' #data
    original_dir = os.path.join(base_dir, dataset_name, 'snapshot_'+str(no_snapshot), 'original_data')
    heterodata_dir = os.path.join(base_dir, dataset_name, 'snapshot_'+str(no_snapshot), 'heterodata')
    heterodata_prev_dir = os.path.join(base_dir, dataset_name, f'snapshot_{(no_snapshot-1)}', 'heterodata')

    mapping_labels = open_pickle(os.path.join(base_dir, dataset_name, 'mapping_labels.pkl'))
    target_type = get_target_type(dataset_name)
    Y_df = pd.read_csv(os.path.join(original_dir, target_type + '_labels.csv'))

    # processing nodes +
    print("processing nodes + mapping...")
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
            K_new_nodes[n_type] = list(mapping[n_type].values())
            K_old_nodes[n_type] = {}

            X = attributes_encoding(df, dataset_name, n_type[:-1], no_snapshot)
            torch.save(X, os.path.join(heterodata_dir, 'features', n_type+'.pt' ))
            #cprint(f'{n_type} features saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
            print(f'{n_type} features saved')

        # mapping labels
        print("mapping labels...")
        #Y['id'] = Y['id'].map(mapping[target_type + 's']) in questo caso non serve
        Y_df['label'] = Y_df['label'].map(mapping_labels)
        Y = torch.tensor(Y_df['label'].values)
        torch.save(Y, os.path.join(heterodata_dir, target_type + '_labels.pt'))
        #cprint(f'Ground truth saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
        print(f'Ground truth saved')

    else:
        mapping = open_pickle(os.path.join(heterodata_prev_dir, 'mapping.pkl'))
        for fname in os.listdir(os.path.join(original_dir, 'nodes')):
            n_type = fname[:-4]  # remove the last 4 characters (".csv")
            print(f"Processing {n_type}")
            map_n_type = mapping[n_type]
            max_id = max(map_n_type.values(), default=-1)
            print(f"Intial max id {max_id}")
            df = pd.read_csv(os.path.join(original_dir, 'nodes', fname))
            id_column = next(col for col in df.columns if "id" in col.lower())
            K_new_nodes[n_type] = []
            for id_str in df[id_column]:
                if id_str not in map_n_type:
                    max_id += 1
                    map_n_type[id_str] = max_id
                K_new_nodes[n_type].append(map_n_type[id_str]) # add to K_new both unseen and changed nodes
            print(f"Final max id {max_id}")
            K_old_nodes[n_type] = [id for id in mapping[n_type].values() if id not in K_new_nodes[n_type]] # add to K_old all - K_new

            X = attributes_encoding(df, dataset_name, n_type[:-1], no_snapshot)
            print(f"Type of X {type(X)}; Shape of X {X.shape}")
            X_prev = torch.load(os.path.join(heterodata_prev_dir, 'features', n_type+'.pt'))
            print(f"Type of X_prev {type(X_prev)}; Shape of X_prev {X_prev.shape}")
            X_ok = torch.zeros(max_id + 1, X_prev.shape[1], dtype=X_prev.dtype)
            #X = torch.cat((X_prev, X), dim=0)

            for id_str, id in map_n_type.items():
                if id_str in df[id_column].values:
                    # NEW: Use row from X if id_str is in df
                    row_index = df[df[id_column] == id_str].index[0]
                    X_ok[id] = X[row_index].cpu()
                else:
                    # OLD: Use row from X_prev if id_str was already in map_n_type and not in df
                    X_ok[id] = X_prev[map_n_type[id_str]].cpu()

            torch.save(X_ok, os.path.join(heterodata_dir, 'features', n_type + '.pt'))
            #cprint(f'{n_type} features saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
            print(f'{n_type} features saved')

        # mapping labels
        print("mapping labels...")
        Y_df['id'] = Y_df['id'].map(mapping[target_type + 's'])
        Y_df['label'] = Y_df['label'].map(mapping_labels)
        Y_prev = torch.load(os.path.join(heterodata_prev_dir, target_type + '_labels.pt'))
        Y_prev_df = pd.DataFrame({"label": Y_prev.tolist()})
        Y_prev_df.reset_index(inplace=True) # Reset the index in Y_prev to simulate the "id" column (0, 1, 2, ...)
        Y_prev_df.columns = ["id", "label"]
        Y_tot = pd.concat([Y_df, Y_prev_df[~Y_prev_df["id"].isin(Y_df["id"])]]) # Remove entries from Y_prev_df if the 'id' already exists in Y_df
        Y_tot = Y_tot.reset_index(drop=True).reset_index() # Reset index to make "id" a sequential integer starting from 0
        Y_tot = Y_tot.rename(columns={"index": "id"})
        Y = torch.tensor(Y_tot["label"].values)
        torch.save(Y, os.path.join(heterodata_dir, target_type + '_labels.pt'))
        #cprint(f'Ground truth saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
        print(f'Ground truth saved')

    # saving mapping, K_new_nodes, K_old_nodes
    save_dict_to_pickle(mapping, os.path.join(heterodata_dir, 'mapping.pkl'))
    save_dict_to_pickle(K_new_nodes, os.path.join(heterodata_dir, 'K_new_nodes.pkl'))
    save_dict_to_pickle(K_old_nodes, os.path.join(heterodata_dir, 'K_old_nodes.pkl'))


    # processing edges
    print("processing edges...")
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
        df = pd.read_csv(f, usecols=['src', 'tgt'])
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
        #cprint(f'{e_type} edgelist saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
        print(f'{e_type} edgelist saved')

    # saving K_new_edges, K_old_edges
    save_dict_to_pickle(K_new_edges, os.path.join(heterodata_dir, 'K_new_edges.pkl'))
    save_dict_to_pickle(K_old_edges, os.path.join(heterodata_dir, 'K_old_edges.pkl'))


dataset_name = "openalex"
snapshots = range(0,7) #0

for snapshot in snapshots:
    heterodata_dir = os.path.join('/mnt/nas/martirano', dataset_name, f'snapshot_{snapshot}', 'heterodata')
    os.makedirs(heterodata_dir, exist_ok=True)
    os.makedirs(os.path.join(heterodata_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(heterodata_dir, 'edgelists'), exist_ok=True)
    extract_knowledge(dataset_name, snapshot)
