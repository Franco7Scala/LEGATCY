import os
import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

from data_preprocessing.openalex import encoding_attributes



def open_pickle(pckl_file):
    file = open(pckl_file, 'rb')
    return pickle.load(file)

def save_dict_to_pickle(data_dict, pckl_file):
    with open(pckl_file, 'wb') as file:
        pickle.dump(data_dict, file)


def get_target_type(dataset_name):
    heterodata_dir = os.path.join('data', dataset_name, 'snapshot_0', 'heterodata')
    fname_labels = next((f for f in os.listdir(heterodata_dir) if f.endswith(".pt")), None)
    return fname_labels.split('_')[0] #xxx_labels.pt

def extract_edge_info(fname):
    #fname_base = fname[:-3]  # remove the last 3 characters (".pt") #if .csv?
    last_dot = fname.rfind(".")
    fname_base = fname[:last_dot]
    first_underscore = fname_base.find("_")  # first occurrence
    last_underscore = fname_base.rfind("_")  # last occurrence
    n_type_src = fname_base[:first_underscore]
    e_type = fname_base[first_underscore + 1:last_underscore]
    n_type_tgt = fname_base[last_underscore + 1:]
    return n_type_src, n_type_tgt, e_type


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


def edges_encoding(df):
    return torch.tensor(df.values.T)


def attributes_encoding(df, dataset_name, n_type):
    tensors = []

    if dataset_name == "openalex":
        df = encoding_attributes(df, n_type) #openalex.py
    elif dataset_name == "mumin":
        print("work in progress")

    for col in df.columns:
        print('##### Processing column ', col, ' #####')
        if df[col].dtype == 'int64':
            tensors.append(torch.tensor(df[col].values, dtype=torch.int32).unsqueeze(1))
        elif df[col].dtype == 'float64':
            tensors.append(torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1))
        elif df[col].dtype == 'bool':
            tensors.append(torch.tensor(df[col].values, dtype=torch.bool).unsqueeze(1))
        elif df[col].dtype == 'object':
            embedding_tensors = df[col].apply(lambda x: torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x)
            embedding_stack = torch.stack(embedding_tensors.tolist())  # Convert to list before stacking
            tensors.append(embedding_stack)
        elif df[col].dtype == 'category':
            enc = LabelEncoder()
            encoded_values = enc.fit_transform(df[col])
            tensors.append(torch.tensor(encoded_values, dtype=torch.int32).unsqueeze(1))
        else:
            raise ValueError(f"Unsupported column type: {df[col].dtype}")

    # Concatenate all tensors along the last dimension
    return torch.cat(tensors, dim=1)


# scaling of numeric columns
def scale_numeric(df, col):
    scaler = MinMaxScaler()
    df[col + '_scaled'] = scaler.fit_transform(df[[col]])
    df.drop([col], axis=1, inplace=True)
    return df


# conversion from datetime64[ns] to timestamp + normalization
# boost: replace nan values with the mean of the column instead of epoch start to avoid otliers (MinMax Scaling is sensitive to outliers)
def convert_datetime_to_timestamp(df, col):
    # df[col+'_timestamp'] = df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else pd.Timestamp('1970-01-01 00:00:00').timestamp())
    mean_date = df[col].mean()
    df[col] = df[col].fillna(mean_date)
    df[col + '_timestamp'] = df[col].apply(lambda x: x.timestamp())
    scaler = MinMaxScaler()
    df[col + '_timestamp_scaled'] = scaler.fit_transform(df[[col + '_timestamp']])
    df.drop([col, col + '_timestamp'], axis=1, inplace=True)
    return df


# encoding of short texts
# boost: apply PCA after SBERT on the individual attribute. Apply the mean to nan values
def encoding_short_text(df, col, target_dim=64):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # df[col+'_encoded'] = df.apply(lambda x: model.encode(x[col]), axis=1)

    # Convert the column to string if it's categorical
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        # if pd.api.types.is_categorical_dtype(df[col]):
        df[col] = df[col].astype(str)

    # Generate embeddings for text entries
    def get_embedding(text_or_list):
        ''' if pd.isna(text_or_list) or text_or_list is None:
            return None '''
        if isinstance(text_or_list, str):
            if text_or_list == "":
                return None
            return model.encode(text_or_list)
        if isinstance(text_or_list, list):
            embeddings = [model.encode(text) for text in text_or_list if text != ""]
            if embeddings:
                return np.mean(embeddings, axis=0)
            return None
        return None

    embeddings = df[col].apply(get_embedding)
    # non_nan_embeddings = embeddings.dropna().values
    non_nan_embeddings = [e for e in embeddings if e is not None]

    if len(non_nan_embeddings) > 0:
        embeddings_matrix = np.vstack(non_nan_embeddings)
        mean_embedding = np.mean(embeddings_matrix, axis=0)

        def replace_missing(embedding):
            if embedding is None:
                return mean_embedding
            else:
                return embedding

        embeddings = embeddings.apply(replace_missing)

    embeddings_matrix = np.vstack(embeddings)
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    df[col + '_encoded'] = [embedding.tolist() for embedding in
                            reduced_embeddings]  # Store the reduced embeddings in a single column as lists
    df.drop([col], axis=1, inplace=True)
    return df


def encoding_long_text(df, col):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    MAX_LENGTH = 384 #modelmax_seq_length

    # Function to create embedding for a given text
    def create_embedding(text):
        if text is None:
            return None
        # Split the text if it's longer than the max length limit
        if len(text.split()) > MAX_LENGTH:
            # Split into chunks within the max token length
            chunks = [text[i:i + MAX_LENGTH] for i in range(0, len(text.split()), MAX_LENGTH)]
            # Compute embeddings for each part and take the mean
            chunk_embeddings = [model.encode(chunk) for chunk in chunks]
            return np.mean(chunk_embeddings, axis=0)
        else:
            # Directly compute embedding if within limit
            return model.encode(text)

    df[col + '_encoded'] = df[col].apply(create_embedding)
    mean_embedding = np.mean([emb for emb in df[col + '_encoded'] if emb is not None], axis=0) # Calculate mean embedding for non-None entries
    df[col + '_encoded'] = df[col + '_encoded'].apply(lambda emb: mean_embedding if emb is None else emb)
    df = df.drop(columns=[col])

    return df


def parsing_embedding_float(df, col):
    df[col + '_parsed'] = df.apply(lambda x: torch.tensor(x[col], dtype=torch.float32), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df


def parsing_embedding_int(df, col):
    df[col + '_parsed'] = df.apply(lambda x: torch.tensor(x[col], dtype=torch.int32), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df


def one_hot_encoding(df, col):
    df[col] = df[col].astype(str)  # added for int values (cluster_labels)
    if df[col].dtype.name == 'category':
        unique_categories = df[col].cat.categories.tolist()
    else:
        unique_categories = df[col].unique().tolist()
    num_categories = len(unique_categories)
    category_to_index = {category: idx for idx, category in enumerate(unique_categories)}

    def one_hot_encode(category):
        category = str(category)  # Ensure the category is treated as a string
        one_hot_vector = np.zeros(num_categories, dtype=int)
        one_hot_vector[category_to_index[category]] = 1
        # return one_hot_vector
        return torch.tensor(one_hot_vector, dtype=torch.int32)

    df[col + '_ohe'] = df.apply(lambda x: one_hot_encode(x[col]), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df



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


