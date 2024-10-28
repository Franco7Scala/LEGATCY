from sys import meta_path

import torch
import os
import time
import numpy
import pandas as pd
import pickle

from torch_geometric.data import HeteroData
from statistics import stdev
from enum import Enum


class Color(Enum):
    EXPERIMENT_CONFIG_INFO = 2
    EXPERIMENT_STATUS_HIGH_PRIORITY = 3
    EXPERIMENT_STATUS_LOW_PRIORITY = 4
    EXPERIMENT_OUTPUT = 6
    WARNING = 5
    OTHER = 7
    BLACK = 8


training_seeds = [123123, 34534534, 21312312, 67678678, 234234234]

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def open_pickle(pckl_file):
    file = open(pckl_file, 'rb')
    return pickle.load(file)


def get_target_type(dataset_name):
    heterodata_dir = os.path.join('data', dataset_name, 'snapshot_0', 'heterodata')
    fname_labels = next((f for f in os.listdir(heterodata_dir) if f.endswith(".pt")), None)
    return fname_labels.split('_')[0] #xxx_labels.pt


def get_metapaths(dataset_name):
    if dataset_name == "openalex":
        metapaths = [[('author', 'paper'), ('paper', 'author')], #APA
             [('author', 'paper'), ('paper', 'is_cited_by', 'paper'), ('paper', 'author')], #APPA
             [('author', 'institution'), ('institution', 'author')]] #AIA
    elif dataset_name == "mumin":
        metapaths = [[('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'is_posted_by', 'user'),
                      ('user', 'posted', 'tweet'),
                      ('tweet', 'discusses', 'claim')],  # CTUTC
                     [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'has_hashtag', 'hashtag'),
                      ('hashtag', 'is_hashtag_of', 'tweet'),
                      ('tweet', 'discusses', 'claim')],  # CTHTC
                     [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'is_replied_by', 'reply'),
                      ('reply', 'reply_to', 'tweet'),
                      ('tweet', 'discusses', 'claim')],  # CTRTC_r
                     [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'is_quoted_by', 'reply'),
                      ('reply', 'quote_of', 'tweet'),
                      ('tweet', 'discusses', 'claim')]]  # CTRTC_q
    else:
        raise ValueError(f"No dataset with name '{dataset_name}'")
    return metapaths


"""
Args --> dataset name (str) + dataset (HetrodataObject)
Returns: --> Target node type (str) + dict (node_type: num_nodes)
"""
def nodes_info(dataset_name, data):
    d = {}
    if dataset_name == "openalex":
        target_type = 'author'
        d['author'] = data['author'].shape[0]
        d['paper'] = data['paper'].shape[0]
        d['institution'] = data['institution'].shape[0]
    elif dataset_name == "mumin":
        target_type = 'claim'
        d['claim'] = data['claim'].shape[0]
        d['tweet'] = data['tweet'].shape[0]
        d['reply'] = data['reply'].shape[0]
        d['user'] = data['user'].shape[0]
        d['hashtag'] = data['hashtag'].shape[0]
        d['article'] = data['article'].shape[0]
        d['image'] = data['image'].shape[0]
    else:
        raise ValueError(f"No dataset with name '{dataset_name}'")
    return target_type, d





"""
Extract a subset of the heterodata object based on a provided mask on nodes ids.
Args --> data (Heterodata): Full dataset; mask (dict): dictionary in the form node_type: bool tensor
Returns: --> Heterodata: New heterodata object
"""
def extract_heterodata_sub(data, mask):
    data_sub = HeteroData()

    # Filter nodes according to the provided mask
    for node_type, mask in mask.items():
        # Apply the mask to the nodes of this type
        data_sub[node_type].x = data[node_type].x[mask]
        # Map old indices to new ones for edge filtering
        index_map = torch.full((data[node_type].num_nodes,), -1, dtype=torch.long)
        index_map[mask] = torch.arange(mask.sum().item())
        # Store the index mapping in the filtered data
        data_sub[node_type].index_map = index_map

        # Filter edges based on the filtered nodes
        for edge_type, edge_index in data.edge_index_dict.items():
            # Split edge_type into (source_type, relation, target_type)
            src_type, _, tgt_type = edge_type
            # Apply the index map to filter edges based on valid source and target nodes
            src_nodes = data_sub[src_type].index_map
            tgt_nodes = data_sub[tgt_type].index_map
            # Filter edges where both nodes exist in the subset
            src_mask = src_nodes[edge_index[0]] != -1
            tgt_mask = tgt_nodes[edge_index[1]] != -1
            valid_edges = src_mask & tgt_mask
            # Apply the mask to the edge index
            filtered_edge_index = edge_index[:, valid_edges]
            filtered_edge_index[0] = src_nodes[filtered_edge_index[0]]
            filtered_edge_index[1] = tgt_nodes[filtered_edge_index[1]]
            # Store the filtered edges in the new data object
            data_sub[edge_type].edge_index = filtered_edge_index

    return data_sub




"""
Computes weights for a multiclass classification task.
Args --> targets (torch.Tensor): A tensor containing the class labels.
Returns: --> torch.Tensor: A tensor of weights, where each weight corresponds to a class.
"""
def compute_weights(targets):
    total_samples = len(targets)
    class_counts = torch.bincount(targets) # Count the occurrences of each class (assuming classes are labeled as 0, 1, 2, ..., n-1)
    weights = total_samples / class_counts.float() # Compute weights inversely proportional to the class frequency
    weights /= weights.sum() ## Normalize the weights so they sum to 1 (optional)

    # Print information for debugging
    for i, count in enumerate(class_counts):
        print(f"Number of class {i}s: {count.item()}")
    print(f"Computed weights: {weights}")

    return weights


#compute mean and standard deviation for multiple runs
def process_metric(df, col):
    data = df[col]
    media = data.mean()
    st_dev = stdev(data.tolist())
    return media, st_dev


# df = pd.DataFrame(columns=['F1_micro', 'F1_macro', 'F1_weighted', 'ROC-AUC'])
def processing_results(df):
    res = pd.DataFrame(columns=df.columns)  # columns=['F1_micro', 'F1_macro', 'F1_weighted', 'ROC-AUC']
    # Compute mean and standard deviation
    for col in df.columns:
        lista = df[col].tolist()
        media, st_dev = process_metric(df, col)
        stringa = "{:0.4f}".format(media) + u"\u00B1" + "{:0.4f}".format(st_dev)
        lista.append(stringa)
        res[col] = lista
    return res

def cprint(text, color=Color.BLACK):
    if color == Color.EXPERIMENT_CONFIG_INFO:
        code_color = "\033[94m"

    elif color == Color.EXPERIMENT_STATUS_HIGH_PRIORITY:
        code_color = "\033[32m"

    elif color == Color.EXPERIMENT_STATUS_LOW_PRIORITY:
        code_color = "\033[92m"

    elif color == Color.WARNING:
        code_color = "\033[91m"

    elif color == Color.EXPERIMENT_OUTPUT:
        code_color = "\033[95m"

    elif color == Color.OTHER:
        code_color = "\033[96m"

    else:
        code_color = "\033[0m"

    print(code_color + str(text) + "\033[0m")


def get_time_in_millis():
    return int(round(time.time() * 1000))


def set_random_seed(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
