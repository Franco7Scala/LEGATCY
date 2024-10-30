import os
import pickle
import torch
from torch_geometric.data import HeteroData



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
        # Map raw indices to new ones for edge filtering
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


