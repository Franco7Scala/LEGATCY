import os
import torch
from collections import Counter

from data.data_utils import open_pickle
from support.utils import get_base_dir

dataset_name = "openalex"
base_dir = os.path.join(get_base_dir(), dataset_name)

mapping_labels_tmp = open_pickle(os.path.join(base_dir, "mapping_labels.pkl")) # str --> int
mapping_labels = dict((v, k) for k, v in mapping_labels_tmp.items()) # int --> str


for snapshot in range(7):
    labels_file = os.path.join(base_dir, f"snapshot_{snapshot}", "heterodata", "author_labels.pt")
    labels = torch.load(labels_file)
    labels_str = [mapping_labels[int(i)] for i in labels]
    distribution = Counter(labels_str)

    print(f"Label Distribution of snapshot {snapshot}:")
    for label, count in distribution.items():
        print(f"{label}: {count}")
    print("##########################")