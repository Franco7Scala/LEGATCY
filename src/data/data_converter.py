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

from support.utils_data import open_pickle, save_dict_to_pickle, get_target_type, extract_edge_info, attributes_encoding, edges_encoding, get_base_dir


#from src.utils import Color, cprint


def extract_knowledge(base_dir, no_snapshot, dataset_name):

    print(f"Processing snapshot {no_snapshot}")
    original_dir = os.path.join(base_dir, f'snapshot_{no_snapshot}', 'original_data') #IN_DIR
    heterodata_dir = os.path.join(base_dir, f'snapshot_{no_snapshot}', 'heterodata') #OUT_DIR
    heterodata_prev_dir = os.path.join(base_dir, f'snapshot_{(no_snapshot-1)}', 'heterodata') #PREV OUT_DIR

    mapping_labels = open_pickle(os.path.join(base_dir, 'mapping_labels.pkl')) #TODO check
    target_type = get_target_type(dataset_name)
    Y_df = pd.read_csv(os.path.join(original_dir, f'{target_type}_labels.csv'))

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

            X = attributes_encoding(df, dataset_name, n_type, no_snapshot)
            torch.save(X, os.path.join(heterodata_dir, 'features', n_type+'.pt' ))
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
            ultimo_max_id_prec = max_id
            print(f"Intial max id {max_id}")
            df = pd.read_csv(os.path.join(original_dir, 'nodes', fname))
            id_column = next(col for col in df.columns if "id" in col.lower())
            K_new_nodes[n_type] = []
            mapping_tmp = []
            for id_str in df[id_column]:
                if id_str not in map_n_type:
                    max_id += 1
                    map_n_type[id_str] = max_id
                    mapping_tmp.append(max_id - ultimo_max_id_prec)

                else:
                    mapping_tmp.append(map_n_type[id_str])

                K_new_nodes[n_type].append(map_n_type[id_str]) # add to K_new both unseen and changed nodes

            print(f"Final max id {max_id}")
            K_old_nodes[n_type] = [id for id in mapping[n_type].values() if id not in K_new_nodes[n_type]] # add to K_old: all - K_new

            X = attributes_encoding(df, dataset_name, n_type, no_snapshot)
            print(f"Type of X {type(X)}; Shape of X {X.shape}")
            X_prev = torch.load(os.path.join(heterodata_prev_dir, 'features', n_type+'.pt'))
            print(f"Type of X_prev {type(X_prev)}; Shape of X_prev {X_prev.shape}")
            #update features (X_ok)
            X_delta_new = torch.zeros((max_id-ultimo_max_id_prec), X_prev.shape[1], dtype=X_prev.dtype)
            X_ok = torch.cat((X_prev, X_delta_new), dim=0)

            if n_type == target_type:
                Y_df['label'] = Y_df['label'].map(mapping_labels)
                Y = torch.tensor(Y_df["label"].values)
                Y_prev = torch.load(os.path.join(heterodata_prev_dir, target_type + '_labels.pt'))
                Y_delta_new = torch.zeros((max_id-ultimo_max_id_prec), dtype=Y_prev.dtype)
                Y_ok = torch.cat((Y_prev, Y_delta_new), dim=0)

            for i in range(X.shape[0]):
                index = mapping_tmp[i]
                X_ok[index] = X[i] #.cpu()

                if n_type == target_type:
                    Y_ok[index] = Y[i]

            torch.save(X_ok, os.path.join(heterodata_dir, 'features', n_type + '.pt'))
            #cprint(f'{n_type} features saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
            print(f'{n_type} features saved')

            if n_type == target_type:
                torch.save(Y_ok, os.path.join(heterodata_dir, target_type + '_labels.pt'))
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
        print(f"processing edge type {e_type}")
        edge_info = extract_edge_info(fname)
        n_type_src = edge_info[0]
        n_type_tgt = edge_info[1]
        f = os.path.join(original_dir, 'edges', fname)
        df = pd.read_csv(f, usecols=['src', 'tgt'])
        print(f"original csv shape: {df.shape}")
        df['src'] = df['src'].map(mapping[n_type_src])
        df['tgt'] = df['tgt'].map(mapping[n_type_tgt])
        print(f"SHAPE WITH NAN: {df.shape}")
        df = df[(df["src"].notna()) & (df["tgt"].notna())].reset_index(drop=True)
        print(f"SHAPE WITHOUT NAN: {df.shape}")

        K_new_edges[e_type] = df[['src', 'tgt']].values.tolist()
        Xe = edges_encoding(df=df)
        print(f"Dimension of Xe {Xe.shape}")

        if no_snapshot==0:
            K_old_edges[e_type] = []
        else:
            old_old_edges = open_pickle(os.path.join(heterodata_prev_dir, 'K_old_edges.pkl'))[e_type]
            old_new_edges = open_pickle(os.path.join(heterodata_prev_dir, 'K_new_edges.pkl'))[e_type]
            K_old_edges[e_type] = old_old_edges + old_new_edges
            Xe_prev = torch.load(os.path.join(heterodata_prev_dir, 'edgelists', e_type+'.pt'))
            print(f"Dimension of Xe_prev {Xe_prev.shape}")
            Xe = torch.cat((Xe_prev, Xe), dim=1)

        torch.save(Xe, os.path.join(heterodata_dir, 'edgelists', e_type + '.pt'))
        #cprint(f'{e_type} edgelist saved', Color.EXPERIMENT_STATUS_LOW_PRIORITY)
        print(f'{e_type} edgelist saved')

    # saving K_new_edges, K_old_edges
    #save_dict_to_pickle(K_new_edges, os.path.join(heterodata_dir, 'K_new_edges.pkl'))
    #save_dict_to_pickle(K_old_edges, os.path.join(heterodata_dir, 'K_old_edges.pkl'))



if __name__ == '__main__':

    dataset_name = "openalex"
    base_dir = f"{get_base_dir()}/{dataset_name}"
    NUM_SNAPSHOTS = range(7)

    for snapshot in NUM_SNAPSHOTS: #STEP 3
        heterodata_dir = os.path.join(base_dir, f'snapshot_{snapshot}', 'heterodata')
        os.makedirs(heterodata_dir, exist_ok=True)
        os.makedirs(os.path.join(heterodata_dir, 'features'), exist_ok=True)
        os.makedirs(os.path.join(heterodata_dir, 'edgelists'), exist_ok=True)
        extract_knowledge(base_dir, snapshot, dataset_name)
