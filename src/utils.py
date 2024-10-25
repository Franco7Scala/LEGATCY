from torch_geometric.data import HeteroData
import torch
import pandas as pd
from statistics import stdev

"""
Build the new heterodata object.
Args --> I, B (dictionaries in the form node_type: ids_list
Returns: --> HeteroData object, including nodes, edges, meta-paths, train_mask, val_mask, test_mask.
"""
def build_new_heterodata(I, B):
    data = HeteroData()
    #.........................
    return data


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







