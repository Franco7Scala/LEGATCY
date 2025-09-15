import os
import torch
import pickle
from collections import Counter

from data.data_utils import open_pickle
from support.utils import get_base_dir



fine_to_coarse = {

    # Artificial Intelligence & Data
    "artificial intelligence": "Artificial Intelligence & Data",
    "machine learning": "Artificial Intelligence & Data",
    "natural language processing": "Artificial Intelligence & Data",
    "computer vision": "Artificial Intelligence & Data",
    "speech recognition": "Artificial Intelligence & Data",
    "data science": "Artificial Intelligence & Data",
    "data mining": "Artificial Intelligence & Data",

    # Systems & Infrastructure
    "operating system": "Systems & Infrastructure",
    "distributed computing": "Systems & Infrastructure",
    "parallel computing": "Systems & Infrastructure",
    "computer hardware": "Systems & Infrastructure",

    # Networks, Web & Security
    "computer network": "Networks, Web & Security",
    "world wide web": "Networks, Web & Security",
    "internet privacy": "Networks, Web & Security",
    "computer security": "Networks, Web & Security",

    # Information & Knowledge
    "database": "Information & Knowledge",
    "information retrieval": "Information & Knowledge",
    "knowledge management": "Information & Knowledge",
    "library science": "Information & Knowledge",

    # Theory & Programming
    "theoretical computer science": "Theory & Programming",
    "programming language": "Theory & Programming",

    # Multimedia & Applications
    "multimedia": "Multimedia & Applications",
    "generic": "Multimedia & Applications",
}


if __name__ == '__main__':

    dataset_name = "openalex"
    base_dir = os.path.join(get_base_dir(), dataset_name)

    mapping_labels_tmp = open_pickle(os.path.join(base_dir, "mapping_labels.pkl"))  # str -> int
    mapping_labels = dict((v, k) for k, v in mapping_labels_tmp.items())  # int -> str

    # unique coarse classes from fine_to_coarse
    coarse_classes = sorted(set(fine_to_coarse.values()))
    coarse_to_int = {cls: idx for idx, cls in enumerate(coarse_classes)}  # str -> int
    int_to_coarse = {idx: cls for cls, idx in coarse_to_int.items()}      # int -> str


    for snapshot in range(7):
        snap_dir = os.path.join(base_dir, f"snapshot_{snapshot}", "heterodata")
        labels_file = os.path.join(snap_dir, "author_labels.pt")

        old_labels_file = os.path.join(snap_dir, "author_labels_all_old.pt")
        if not os.path.exists(old_labels_file):
            os.rename(labels_file, old_labels_file)

        # Load old labels
        labels_all = torch.load(old_labels_file)  # tensor of ints
        labels_str = [mapping_labels[int(i)] for i in labels_all]

        # Map fine -> coarse
        labels_coarse_str = [fine_to_coarse[lbl] for lbl in labels_str]
        labels_coarse_int = torch.tensor([coarse_to_int[lbl] for lbl in labels_coarse_str])

        torch.save(labels_coarse_int, labels_file)

        # Print distribution for sanity check
        dist = Counter(labels_coarse_str)
        print(f"Snapshot {snapshot} new distribution:")
        for label, count in dist.items():
            print(f"{label}: {count}")
        print("#########################################")


    # Replace mapping_labels.pkl
    old_map_file = os.path.join(base_dir, "mapping_labels.pkl")
    backup_map_file = os.path.join(base_dir, "mapping_labels_all_old.pkl")
    if not os.path.exists(backup_map_file):
        os.rename(old_map_file, backup_map_file)

    # Save new int -> coarse mapping
    with open(old_map_file, "wb") as f:
        pickle.dump(int_to_coarse, f)

    # Save detailed fine -> coarse mapping
    detailed_map_file = os.path.join(base_dir, "mapping_labels_detailed.pkl")
    with open(detailed_map_file, "wb") as f:
        pickle.dump(fine_to_coarse, f)

