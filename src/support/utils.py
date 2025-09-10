from sys import meta_path

import torch
import os
import time
import numpy
import pandas as pd
from statistics import stdev
from enum import Enum
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from deprecated import deprecated


class Color(Enum):
    EXPERIMENT_CONFIG_INFO = 2
    EXPERIMENT_STATUS_HIGH_PRIORITY = 3
    EXPERIMENT_STATUS_LOW_PRIORITY = 4
    EXPERIMENT_OUTPUT = 6
    WARNING = 5
    OTHER = 7
    BLACK = 8


training_seeds = [123123, 34534534, 21312312, 67678678, 234234234]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")     #"cpu"#


def get_base_dir():
    #return '/home/martirano/data'
    return '/home/scala/projects/GNN_ContinualLerning/data'
    #return '/home/scala/datasets/mumin'


def count_n_snapshots(dataset_name):
    return len([f.path for f in os.scandir(f"{get_base_dir()}/{dataset_name}") if f.is_dir() and "snapshot_" in f.name])


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
    for i, count in enumerate(class_counts):
        print(f"Number of class {i}: {count.item()}")
    print(f"Computed weights: {weights}")

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
