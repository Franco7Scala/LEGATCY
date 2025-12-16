import torch
import os
import time
import numpy
import pandas as pd
import copy

from sys import meta_path
from statistics import stdev
from enum import Enum
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from deprecated import deprecated
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import to_hetero


class Color(Enum):
    EXPERIMENT_CONFIG_INFO = 2
    EXPERIMENT_STATUS_HIGH_PRIORITY = 3
    EXPERIMENT_STATUS_LOW_PRIORITY = 4
    EXPERIMENT_OUTPUT = 6
    WARNING = 5
    OTHER = 7
    BLACK = 8


class Kwargs:
    pass


training_seeds = [123123, 34534534, 21312312, 67678678, 234234234]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")     #"cpu"#


def get_base_dir():
    return '/home/martirano/data'
    #return '/home/scala/projects/GNN_ContinualLearning/data'


def get_metapaths(dataset_name):
    if dataset_name == "openalex":
        metapaths = [[('author', 'writes', 'paper'),
                      ('paper', 'is_written_by', 'author')],  # APA
                     [('author', 'is_affiliated_with', 'institution'),
                      ('institution', 'is_affiliation_of', 'author')], #AIA
                     [('author', 'writes', 'paper'),
                      ('paper', 'cites', 'paper'),
                      ('paper', 'is_written_by', 'author')]] #APPA

    elif dataset_name.lower() == "imdb":
        metapaths = [[('movie', 'to', 'actor'),
                      ('actor', 'to', 'movie')], # MAM
                     [('movie', 'to', 'director'),
                      ('director', 'to', 'movie')]] #MDM

    elif dataset_name.lower() == "dblp":
        metapaths = [[('author', 'to', 'paper'),
                      ('paper', 'to', 'author')], # APA
                     [('author', 'to', 'paper'),
                      ('paper', 'to', 'conference'),
                      ('conference', 'to', 'paper'),
                      ('paper', 'to', 'author')], #APCPA
                     [('author', 'to', 'paper'),
                      ('paper', 'to', 'term'),
                      ('term', 'to', 'paper'),
                      ('paper', 'to', 'author')]] #APTPA

    elif dataset_name.lower() == "aminer":
        metapaths = [[('author', 'writes', 'paper'),
                      ('paper', 'written_by', 'author')],  # APA
                     [('author', 'writes', 'paper'),
                      ('paper', 'published_in', 'venue'),
                      ('venue', 'publishes', 'paper'),
                      ('paper', 'written_by', 'author')]]  # APVPA

    elif dataset_name.lower() == "politifact":
        metapaths = [
            [('news', 'is_discussed_by', 'tweet'),
             ('tweet', 'is_posted_by', 'user'),
             ('user', 'posted', 'tweet'),
             ('tweet', 'discusses', 'news')],  # NTUTN
            [('news', 'is_discussed_by', 'tweet'),
             ('tweet', 'has_hashtag', 'hashtag'),
             ('hashtag', 'is_hashtag_of', 'tweet'),
             ('tweet', 'discusses', 'news')],  # NTHTN
            [('news', 'is_discussed_by', 'tweet'),
             ('tweet', 'is_posted_by', 'user'),
             ('user', 'mentions', 'user'),
             ('user', 'posted', 'tweet'),
             ('tweet', 'discusses', 'news')]]  # NTUUTN

    elif dataset_name.lower() == "mumin":
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
        raise Exception("no metapaths defined for this dataset! Cretina!")

    return metapaths


def count_n_snapshots(dataset_name):
    return len([f.path for f in os.scandir(f"{get_base_dir()}/{dataset_name}") if f.is_dir() and "snapshot_" in f.name])


def build_heterodata(dataset_name, target_type):
    data = HeteroData()
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, "heterodata")
    feats_dir = os.path.join(heterodata_dir, "features")
    edges_dir = os.path.join(heterodata_dir, "edgelists")

    # nodes
    files_feats = [f for f in os.listdir(feats_dir) if os.path.isfile(os.path.join(feats_dir, f))]
    for fname in files_feats:
        n_type = fname[:-3]  # remove the last 4 characters (".pt")
        data[n_type].x = torch.load(os.path.join(feats_dir, fname))

    # ground truth for target_type
    data[target_type].y = torch.load(os.path.join(heterodata_dir, f'{target_type}_labels.pt'))

    # edges
    files_edges = [f for f in os.listdir(edges_dir) if os.path.isfile(os.path.join(edges_dir, f))]
    for fname in files_edges:
        n_type_src, n_type_tgt, e_type = _extract_edge_info(fname)
        edge_index = torch.load(os.path.join(heterodata_dir, "edgelists", fname))
        if edge_index.dtype == torch.float64:
            edge_index = edge_index.to(torch.int64)
        data[n_type_src, e_type, n_type_tgt].edge_index = edge_index

    transform = T.RandomNodeSplit(num_val=0, num_test=0.30)  # train-val-test split: 70-0-30
    data = transform(data)

    return data

def _extract_edge_info(fname):
    #fname_base = fname[:-3]  # remove the last 3 characters (".pt") #if .csv?
    last_dot = fname.rfind(".")
    fname_base = fname[:last_dot]
    first_underscore = fname_base.find("_")  # first occurrence
    last_underscore = fname_base.rfind("_")  # last occurrence
    n_type_src = fname_base[:first_underscore]
    e_type = fname_base[first_underscore + 1:last_underscore]
    n_type_tgt = fname_base[last_underscore + 1:]
    return n_type_src, n_type_tgt, e_type


def to_categorical(data, n_classes):
    result = numpy.zeros((data.size, n_classes), dtype=int)
    result[numpy.arange(data.size), data] = 1
    return result


def compute_auc(y_true, y_pred):
    all_classes = numpy.arange(y_pred.shape[1])
    scores = roc_auc_score(y_true=label_binarize(y_true, classes=all_classes), y_score=y_pred, average=None, multi_class="ovo")
    valid_scores = scores[~numpy.isnan(scores)]
    return numpy.mean(valid_scores)


"""
Computes weights for a multiclass classification task.
Args --> targets (torch.Tensor): A tensor containing the class labels.
Returns: --> torch.Tensor: A tensor of weights, where each weight corresponds to a class.
"""
@deprecated(reason="Skipped to focal loss!")
def compute_weights(targets):
    total_samples = len(targets)
    class_counts = torch.bincount(targets.to(torch.int64)) # Count the occurrences of each class (assuming classes are labeled as 0, 1, 2, ..., n-1)
    weights = total_samples / class_counts.float() # Compute weights inversely proportional to the class frequency
    weights /= weights.sum() ## Normalize the weights so they sum to 1 (optional)

    # Print information for debugging
    # for i, count in enumerate(class_counts):
    #     print(f"Number of class {i}: {count.item()}")
    # print(f"Computed weights: {weights}")

    return weights


def get_class_distribution(data, target_type):
    targets = data[target_type].y
    return torch.bincount(targets.to(torch.int64))

    
#compute mean and standard deviation for multiple runs
def process_metric(df, col):
    data = df[col]
    media = data.mean()
    st_dev = stdev(data.tolist()) if len(data.tolist()) > 1 else 0
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


def predictable_hash(text:str):
  hash = 0
  for ch in text:
    hash = ( hash * 281 ^ ord(ch) * 997 ) & 0xFFFFFFFF

  return hash


def merge_masks(masks):
    if len(masks) == 0:
        return []

    if len(masks) == 1:
        masks[0]

    merged_mask = copy.deepcopy(masks[0])
    for mask in masks[1:]:
        for key in merged_mask:
            merged_mask[key] = torch.cat((merged_mask[key], mask[key]), dim=0)

    return merged_mask


def print_samples_count(nodes_dict):
    for node_type in nodes_dict:
        cprint(f"- Class {node_type}: {len(nodes_dict[node_type])} samples", Color.EXPERIMENT_CONFIG_INFO)

def str2bool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True

    elif val in ("n", "no", "f", "false", "off", "0"):
        return False

    else:
        raise ValueError(f"invalid truth value {val}")
