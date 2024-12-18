import os
import pickle
import torch
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

from src import utils


def get_base_dir():
    #return '/home/scala/projects/GNN_ContinualLerning/data'
    #return '/mnt/nas/martirano'  #data
    return '/home/martirano/data'

def open_pickle(pckl_file):
    file = open(pckl_file, 'rb')
    return pickle.load(file)

def save_dict_to_pickle(data_dict, pckl_file):
    with open(pckl_file, 'wb') as file:
        pickle.dump(data_dict, file)


def get_target_type(dataset_name):
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, 'snapshot_0', 'heterodata')
    fname_labels = next((f for f in os.listdir(heterodata_dir) if f.endswith(".pt")), None)
    if fname_labels is None:
        if dataset_name == "openalex":
            return "author"
        elif dataset_name == "mumin":
            return "claim"
        else:
            raise ValueError(f"No dataset with name '{dataset_name}'")
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
    if "openalex" in dataset_name:
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


def get_openalex_sub_concepts_list():
    return ["multimedia", "database", "internet privacy", "natural language processing", "data science",
                    "artificial intelligence", "distributed computing", "computer hardware",
                    "theoretical computer science", "library science", "operating system", "world wide web",
                    "parallel computing", "information retrieval", "computer security", "knowledge management",
                    "computer vision", "data mining", "speech recognition", "programming language",
                    "computer network", "machine learning"] #"computer architecture", "real time computing", "computer graphics images", "human computer interaction",

def get_ohe_types(dataset_name, n_type, col):
    values = []
    for i in range(7):
        df = pd.read_csv(os.path.join(get_base_dir(), f"{dataset_name}/snapshot_{i}/original_data/nodes/{n_type}s.csv"))
        values.extend(df[col].fillna("unknown").drop_duplicates().tolist())
    return list(set(values))


def edges_encoding(df):
    #print(df.dtypes)
    return torch.tensor(df.values.T)



def attributes_encoding(df, dataset_name, n_type, no_snapshot):
    print(f"Processing {n_type}")
    columns_ok = []
    df_ok = None
    #if dataset_name == "openalex":
    if "openalex" in dataset_name:
        #df = pd.read_csv(os.path.join(get_base_dir(), dataset_name, f'snapshot_{no_snapshot}', f'original_data/nodes/{n_type}s.csv'))
        if n_type == "author":
            df_ok = df[['name', 'n_works', 'n_cit', 'impact_factor', 'h_index', 'i10_index']].copy()
            for col in df_ok.columns.tolist():
                if col == 'name':
                    df_ok = encoding_short_text(df_ok, col, target_dim=64)
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda lista: [float(x) for x in lista])
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda x: torch.tensor(x))
                    print(f"Column {col + '_encoded'} of type {df_ok[col + '_encoded'].dtype} with elements of type {type(df_ok[col + '_encoded'].tolist()[0])}")
                    columns_ok.append(col+'_encoded')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
                else:
                    df_ok = scale_numeric(df_ok, col)
                    print(f"Column {col + '_scaled'} of type {df_ok[col + '_scaled'].dtype} with elements of type {type(df_ok[col + '_scaled'].tolist()[0])}")
                    columns_ok.append(col + '_scaled')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
        elif n_type == "institution":
            df_ok = df[['name', 'country-code', 'type']].copy()
            for col in df_ok.columns.tolist():
                if col == 'name':
                    df_ok = encoding_short_text(df_ok, col, target_dim=64)
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda lista: [float(x) for x in lista])
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda x: torch.tensor(x))
                    print(f"Column {col + '_encoded'} of type {df_ok[col + '_encoded'].dtype} with elements of type {type(df_ok[col + '_encoded'].tolist()[0])}")
                    columns_ok.append(col + '_encoded')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
                elif col == 'country-code':
                    df_ok[col] = df_ok[col].fillna("unknown")
                    df_ok = one_hot_encoding(dataset_name, n_type, df_ok, col)
                    print(f"Column {col + '_ohe'} of type {df_ok[col + '_ohe'].dtype} with elements of type {type(df_ok[col + '_ohe'].tolist()[0])}")
                    columns_ok.append(col + '_ohe')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
                else:  # type
                    df_ok[col] = df_ok[col].fillna("unknown")
                    df_ok[col] = df_ok[col].astype('category')
                    print(f"Column {col} of type {df_ok[col].dtype} with elements of type {type(df_ok[col].tolist()[0])}")
                    columns_ok.append(col)
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
        elif n_type == "paper":
            df_ok = df[['title', 'num_citations', 'abstract', 'filtered_concepts']].copy()
            for col in df_ok.columns.tolist():
                if col == 'title':
                    df_ok = encoding_short_text(df_ok, col, target_dim=128)
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda lista: [float(x) for x in lista])
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda x: torch.tensor(x))
                    print(f"Column {col+'_encoded'} of type {df_ok[col+'_encoded'].dtype} with elements of type {type(df_ok[col+'_encoded'].tolist()[0])}")
                    columns_ok.append(col+'_encoded')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
                elif col == "abstract":
                    df_ok = encoding_long_text(df_ok, col)
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda lista: [float(x) for x in lista])
                    df_ok[col+'_encoded'] = df_ok[col+'_encoded'].apply(lambda x: torch.tensor(x))
                    print(f"Column {col+'_encoded'} of type {df_ok[col + '_encoded'].dtype} with elements of type {type(df_ok[col + '_encoded'].tolist()[0])}")
                    columns_ok.append(col + '_encoded')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
                elif col == 'filtered_concepts':
                    col = "concepts"
                    df_ok.rename(columns={'filtered_concepts': col}, inplace=True)
                    df_ok = one_hot_encoding_list(df_ok, col, get_openalex_sub_concepts_list())
                    print(f"Column {col+'_ohe'} of type {df_ok[col + '_ohe'].dtype} with elements of type {type(df_ok[col + '_ohe'].tolist()[0])}")
                    columns_ok.append(col + '_ohe')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
                else:  # 'num_citations:
                    df_ok = scale_numeric(df_ok, col)
                    print(f"Column {col+'_scaled'} of type {df_ok[col + '_scaled'].dtype} with elements of type {type(df_ok[col + '_scaled'].tolist()[0])}")
                    columns_ok.append(col + '_scaled')
                    print(f"Processed {col}: columns are now {df_ok.columns.tolist()}")
        else:
            raise ValueError(f"Unsupported node type: {n_type}")
    elif dataset_name == "mumin":
        print("work in progress")

    tensors = []
    print(columns_ok)
    print(df_ok.columns.tolist())
    for col in columns_ok:
        print(col, df_ok[col].dtype)
        print('##### Processing column ', col, ' #####')
        if df_ok[col].dtype == 'int64':
            tensors.append(torch.tensor(df_ok[col].values, dtype=torch.int32).unsqueeze(1))
        elif df_ok[col].dtype == 'float64':
            tensors.append(torch.tensor(df_ok[col].values, dtype=torch.float32).unsqueeze(1))
        elif df_ok[col].dtype == 'bool':
            tensors.append(torch.tensor(df_ok[col].values, dtype=torch.bool).unsqueeze(1))
        elif df_ok[col].dtype == 'object':
            embedding_tensors = df_ok[col].apply(lambda x: torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x)
            embedding_stack = torch.stack(embedding_tensors.tolist())  # Convert to list before stacking
            tensors.append(embedding_stack)
        elif df_ok[col].dtype == 'category':
            enc = LabelEncoder()
            encoded_values = enc.fit_transform(df_ok[col])
            tensors.append(torch.tensor(encoded_values, dtype=torch.int32).unsqueeze(1))
        else:
            raise ValueError(f"Unsupported column type: {df_ok[col].dtype}")

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
    device = utils.get_device()
    print(f"Device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
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
    #TODO per dataset intero
    #pca = PCA(n_components=min(target_dim, embeddings_matrix.shape[0])) #serve per il subset #'mle'.
    #reduced_embeddings = pca.fit_transform(embeddings_matrix)
    #df[col + '_encoded'] = [embedding.tolist() for embedding in reduced_embeddings]  # Store the reduced embeddings in a single column as lists
    df[col + '_encoded'] = [embedding.tolist() for embedding in
                            embeddings_matrix]  # Store the reduced embeddings in a single column as lists
    df.drop([col], axis=1, inplace=True)
    return df


def encoding_long_text(df, col):
    device = utils.get_device()
    print(f"Device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    MAX_LENGTH = 384  # Model's max sequence length

    # Function to create embedding for a given text
    def create_embedding(text):
        if text is None:
            return None
        # Split the text into chunks within the max token length
        if len(text.split()) > MAX_LENGTH:
            chunks = [text[i:i + MAX_LENGTH] for i in range(0, len(text.split()), MAX_LENGTH)]
            # Compute embeddings for each chunk and take the mean
            chunk_embeddings = [model.encode(chunk) for chunk in chunks]
            return np.mean(chunk_embeddings, axis=0)
        else:
            # Directly compute embedding if within limit
            return model.encode(text)

    # Process the DataFrame in batches to manage memory usage
    batch_size = 32  # Adjust batch size based on available memory
    encoded_values = []

    for i in range(0, len(df), batch_size):
        batch = df[col].iloc[i:i + batch_size]
        batch_embeddings = batch.apply(create_embedding)
        encoded_values.extend(batch_embeddings)

        # Clear the CUDA cache to free up memory
        torch.cuda.empty_cache()

    df[col + '_encoded'] = encoded_values
    # Calculate mean embedding for non-None entries
    mean_embedding = np.mean([emb for emb in df[col + '_encoded'] if emb is not None], axis=0)
    # Replace None values with the mean embedding
    df[col + '_encoded'] = df[col + '_encoded'].apply(lambda emb: mean_embedding if emb is None else emb)
    # Drop the original column
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


#get_ohe_types(dataset_name, n_type, col)
def one_hot_encoding(dataset_name, n_type, df, col):
    df[col] = df[col].astype(str)  # added for int values (cluster_labels)
    """"
    if df[col].dtype.name == 'category':
        unique_categories = df[col].cat.categories.tolist()
    else:
        unique_categories = df[col].unique().tolist()
    """
    unique_categories = get_ohe_types(dataset_name, n_type, col)
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


def one_hot_encoding_list(df, col, values):
    df[col] = df[col].astype(str) #Ensure the target column is in string format
    num_values = len(values)
    values_dict = {val: idx for idx, val in enumerate(values)}

    def one_hot_encode(lista):
        one_hot_vector = np.zeros(num_values, dtype=int) # Initialize a zero vector for the number of concepts
        # Set the index of each present concept to 1
        for elem in eval(lista):
            elem_lower = elem[1].lower()
            if elem_lower in values_dict:
                one_hot_vector[values_dict[elem_lower]] = 1
        return torch.tensor(one_hot_vector, dtype=torch.int32)

    # Apply the one-hot encoding function to each row and create a new column
    df[col + '_ohe'] = df.apply(lambda x: one_hot_encode(x[col]), axis=1)
    df.drop([col], axis=1, inplace=True)
    return df
