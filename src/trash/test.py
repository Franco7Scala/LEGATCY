import os
import torch
import pandas as pd

from data_utils import get_base_dir


dataset_name = "potato_openalex"
no_snapshot = 1
heterodata_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{no_snapshot}', 'heterodata')
original_data_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{no_snapshot}', 'original_data')


authors = torch.load(os.path.join(heterodata_dir, 'features', 'authors.pt'))
print(f"Num author nodes: {authors.shape[0]}")
labels = torch.load(os.path.join(heterodata_dir, 'author_labels.pt'))
print(f"Num author labels: {labels.shape[0]}")


""" check from dataframes """

authors_new = pd.read_csv(os.path.join(original_data_dir, 'nodes', 'authors.csv'))['id'].tolist()
labels_new = pd.read_csv(os.path.join(original_data_dir, 'author_labels.csv'))['id'].tolist()

authors_old = []
labels_old = []
for i in range(no_snapshot):
    prev_original_data_dir = os.path.join(get_base_dir(), dataset_name, f'snapshot_{i}', 'original_data')
    authors_old_others = pd.read_csv(os.path.join(prev_original_data_dir, 'nodes', 'authors.csv'))['id'].tolist()
    authors_old = list(set(authors_old + authors_old_others))
    labels_old_others = pd.read_csv(os.path.join(prev_original_data_dir, 'author_labels.csv'))['id'].tolist()
    labels_old = list(set(labels_old + labels_old_others))

authors_all = list(set(authors_old + authors_new))
print(f"Num author nodes from df: {len(authors_all)}")

labels_all = list(set(labels_old + labels_new))
print(f"Num author labels from df: {len(labels_all)}")

