import os
import pandas as pd
import numpy as np
import torch
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import hdbscan #from paper "Accelerated Hierarchical Density Based Clustering"
import re

from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch_geometric.transforms as T

from utils import get_device

#import emoji

dir_base = '/mnt/nas/guarascio/fakenews_datasets/Politifact/politifact_in_mumin_format/'
output_dir = "/mnt/nas/martirano/politifact_cleaned"

def count_nan_or_empty(series):
    #return series.apply(lambda x: pd.isna(x) or (isinstance(x, list) and len(x) == 0)).sum()
    return series.apply(lambda x: pd.isna(x) or (isinstance(x, str) and (x == '[]' or x == 'none'))).sum()

def drop_columns_with_many_nans(df):
    threshold = len(df) / 2
    columns_to_drop = [col for col in df.columns if count_nan_or_empty(df[col]) > threshold]
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def drop_columns_starting_with(df, prefix):
    columns_to_drop = [col for col in df.columns if col.startswith(prefix)]
    df_cleaned = df.drop(columns=columns_to_drop)
    return df_cleaned

def extract_base_source_news(url):
    if url is None:
        return "unknown"
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    # Extract second-level domain (e.g., cnn, nytimes)
    if len(domain_parts) >= 2:
        return domain_parts[-2]
    return "unknown"

def extract_base_source_tweet(source):
    match = re.search(r'>(.*?)<', source)
    return match.group(1) if match else "Unknown"

def label_encoding(df, col):
    label_encoder = LabelEncoder()
    encoded_data = label_encoder.fit_transform(df[col])
    # original_data = label_encoder.inverse_transform(encoded_data)
    #print(original_data)
    return encoded_data

# scaling of numeric columns
def scale_numeric(df, col):
    scaler = MinMaxScaler()
    df[col+'_scaled'] = scaler.fit_transform(df[[col]])
    df.drop([col], axis=1, inplace=True)
    return df

def convert_to_tensor(s):
    # array = ast.literal_eval(array_string)
    # return torch.tensor(array, dtype=torch.float32)
    arr = np.fromstring(s.strip('[]'), sep=' ')
    return torch.tensor(arr, dtype=torch.float32)

# encoding of short texts
# boost: apply PCA after SBERT on the individual attribute. Apply the mean to nan values
def encoding_short_text(df, col, target_dim=28):
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

def clustering(df, col):
    threshold = 0.75
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=1-threshold)
    labels = clusterer.fit_predict(df[col].tolist())
    df['cluster_label'] = labels
    return df

def outliers_remapping(df, col):
    new_values = []
    new_value_counter = df[col].max()+1
    for value in df[col]:
        if value == -1:
            new_values.append(new_value_counter)
            new_value_counter += 1
        else:
            new_values.append(value)

    df[col+'_no_outliers'] = new_values
    return df

def clean_text(text):
    text = re.sub('RT @[A-Za-z0-9_]+:', '', text)  # retweet
    text = re.sub('@[A-Za-z0-9_]+', '', text)  # tag - users
    text = re.sub('https?://\S+|www\.\S+', '', text)  # url
    text = re.sub('#', '', text)  # hashtags
    #text = emoji.replace_emoji(text, replace='') #emoji
    return text


def safe_tensor_conversion(x):
    try:
        return torch.tensor(x, dtype=torch.float32)
    except TypeError:
        print(f"Skipping non-numeric value: {x}")
        return torch.tensor(0.0, dtype=torch.float32)

def encoding_attributes(df):
    tensors = []
    for col in df.columns:
        print('##### Processing column ', col, ' #####')
        if df[col].dtype == 'int64':
            tensors.append(torch.tensor(df[col].values, dtype=torch.int32).unsqueeze(1))
        elif df[col].dtype == 'float64':
            tensors.append(torch.tensor(df[col].values, dtype=torch.float32).unsqueeze(1))
        elif df[col].dtype == 'bool':
            tensors.append(torch.tensor(df[col].values, dtype=torch.bool).unsqueeze(1))
        elif df[col].dtype == 'object':
            embedding_tensors = df[col].apply(lambda x: safe_tensor_conversion(x) if not isinstance(x, torch.Tensor) else x)
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


def edges_encoding(df):
    return torch.tensor(df.values.T)

def edges_rev_encoding(df):
    df_rev = df[['tgt', 'src']]
    return torch.tensor(df_rev.values.T)


""" NEWS """
'''
df_N = pd.read_csv(os.path.join(dir_base, 'news_politifact.csv'))
df_N_ok = drop_columns_with_many_nans(df_N)
print(df_N_ok.shape)
df_N_ok = drop_columns_starting_with(df_N_ok, 'meta_data')
print(df_N_ok.shape)
df_N_ok['base_source'] = df_N_ok["source"].apply(lambda x: extract_base_source_news(x) if pd.notna(x) else "unknown")
df_N_ok.drop(columns=["images", "top_img", "url", "source", "canonical_link", "news_id"], index=1, inplace=True)
print(df_N_ok.shape)
print(df_N_ok.columns)
df_N_ok.to_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"), index=False)
'''
'''
print("processing NEWS nodes")
df_N = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "news.csv"))
print("Encoding text...")
df_N = encoding_short_text(df_N, "text", target_dim=256)
print("Encoding title...")
df_N = encoding_short_text(df_N, "title", target_dim=128)
print("Encoding source...")
df_N = encoding_short_text(df_N, "base_source", target_dim=64)
NX = encoding_attributes(df_N[['title_encoded', 'text_encoded', 'base_source_encoded']].copy())
print(NX.shape)
torch.save(NX, os.path.join(output_dir, "heterodata", "features", "NX_tensor.pt"))
N_labels = label_encoding(df_N, 'label').tolist()
NY = torch.tensor(N_labels)
torch.save(NY, os.path.join(output_dir, "heterodata", "NY_tensor.pt"))
'''

""" USER """
'''
df_U = pd.read_csv(os.path.join(dir_base, 'user_politifact.csv'))
df_U_ok = drop_columns_with_many_nans(df_U)
print(df_U_ok.shape)
df_U_ok.drop(columns=["is_translation_enabled", "screen_name", "user_id"], index=1, inplace=True)
print(df_U_ok.shape)
print(df_U_ok.columns)
df_U_ok.to_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"), index=False)
'''
'''
print("Processing USER nodes")
df_U = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "user.csv"))
print("Encoding location...")
df_U = encoding_short_text(df_U, "location", target_dim=128)
print("Encoding description...")
df_U = encoding_short_text(df_U, "description", target_dim=128)
df_U = scale_numeric(df_U, 'followers_count')
df_U = scale_numeric(df_U, 'friends_count')
df_U = scale_numeric(df_U, 'listed_count')
df_U = scale_numeric(df_U, 'favourites_count')
df_U = scale_numeric(df_U, 'statuses_count')
df_U = scale_numeric(df_U, 'created_at')
UX = encoding_attributes(df_U)
print(UX.shape)
torch.save(UX, os.path.join(output_dir, "heterodata", "features", "UX_tensor.pt"))
'''

""" HASHTAG """
'''
df_H = pd.read_csv(os.path.join(dir_base, 'hashtag_politifact.csv'))
df_H.drop(columns=["hashtag_id"], index=1, inplace=True)
print(df_H.shape)
df_H.to_csv(os.path.join(output_dir, "original_data", "nodes", "hashtag.csv"), index=False)
'''
'''
df_H = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "hashtag.csv"))
print("Encoding texts...")
df_H = encoding_short_text(df_H, "hashtag", target_dim=128)
print("Performing clustering...")
df_H = clustering(df_H, "hashtag_encoded")
num_clusters = df_H['cluster_label'].unique().shape[0]-1
num_outliers = df_H[df_H['cluster_label']== -1].shape[0]
print(f"Number of clusters: {num_clusters}")
print(f"Number of outliers: {num_outliers}")
df_H = outliers_remapping(df_H, "cluster_label")
HX = encoding_attributes(df_H[['hashtag_encoded', 'cluster_label_no_outliers']].copy())
print(HX.shape)
torch.save(HX, os.path.join(output_dir, "heterodata", "features", "HX_tensor.pt"))
'''

""" TWEET """
'''
#df_T_ok = pd.read_csv(os.path.join(dir_base, 'tweet_embedding_politifact_with_bot.csv'))
df_T_ok = pd.read_csv(os.path.join("/mnt/nas/scala", "tweet_embedding_politifact_with_bot.csv"))
#print(df_T.shape)
#df_T_ok = drop_columns_with_many_nans(df_T)
print(df_T_ok.shape)
df_T_ok.drop(columns=["tweet_id", "source", "Unnamed: 0"], index=1, inplace=True)
df_T_ok.to_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"), index=False)
print(df_T_ok.columns)
'''

#print("Processing TWEET nodes")
'''
df_T = pd.read_csv(os.path.join(output_dir, "original_data", "nodes", "tweet.csv"), )
df_T["text_emb_twhin_bert_base"] = df_T["text_emb_twhin_bert_base"].apply(convert_to_tensor)
df_T.drop(columns = ["text"], axis=1, inplace=True)
#df_T = encoding_short_text(df_T, "text", target_dim=384)
df_T['lang'] = df_T['lang'].astype('category')
df_T = scale_numeric(df_T, 'created_at')
df_T = scale_numeric(df_T, 'retweet_count')
df_T = scale_numeric(df_T, 'favorite_count')
print(df_T.columns)
TX = encoding_attributes(df_T)
print(TX.shape)
torch.save(TX, os.path.join(output_dir, "heterodata", "features", "TX_tensor_v2.pt"))
'''

""" EDGES """
'''
df_TN = pd.read_csv(os.path.join(dir_base, "tweet_discusses_news_id_politifact.csv"))
df_TN.to_csv(os.path.join(output_dir, "original_data", "edges", "tweet_discusses_news.csv"), index=False)
TN_tensor = edges_encoding(df_TN)
NT_tensor = edges_rev_encoding(df_TN)
torch.save(TN_tensor, os.path.join(output_dir, "heterodata", "edgelists", "tweet_discusses_news.pt"))
torch.save(NT_tensor, os.path.join(output_dir, "heterodata", "edgelists", "news_is_discussed_by_tweet.pt"))

df_TH = pd.read_csv(os.path.join(dir_base, "tweet_has_hashtag_id_politifact.csv"))
df_TH.to_csv(os.path.join(output_dir, "original_data", "edges", "tweet_has_hashtag_hashtag.csv"), index=False)
TH_tensor = edges_encoding(df_TH)
HT_tensor = edges_rev_encoding(df_TH)
torch.save(TH_tensor, os.path.join(output_dir, "heterodata", "edgelists", "tweet_has_hashtag_hashtag.pt"))
torch.save(HT_tensor, os.path.join(output_dir, "heterodata", "edgelists", "hashtag_is_hashtag_of_tweet.pt"))

df_UT = pd.read_csv(os.path.join(dir_base, "user_posted_tweet_id_politifact.csv"))
df_UT.to_csv(os.path.join(output_dir, "original_data", "edges", "user_posted_tweet.csv"), index=False)
UT_tensor = edges_encoding(df_UT)
TU_tensor = edges_rev_encoding(df_UT)
torch.save(UT_tensor, os.path.join(output_dir, "heterodata", "edgelists", "user_posted_tweet.pt"))
torch.save(TU_tensor, os.path.join(output_dir, "heterodata", "edgelists", "tweet_is_posted_by_user.pt"))

df_UR = pd.read_csv(os.path.join(dir_base, "user_retweeted_tweet_id_politifact.csv"))
df_UR.to_csv(os.path.join(output_dir, "original_data", "edges", "user_retweeted_tweet.csv"), index=False)
UR_tensor = edges_encoding(df_UR)
RU_tensor = edges_rev_encoding(df_UR)
torch.save(UR_tensor, os.path.join(output_dir, "heterodata", "edgelists", "user_retweeted_tweet.pt"))
torch.save(RU_tensor, os.path.join(output_dir, "heterodata", "edgelists", "tweet_is_retweeted_by_user.pt"))

df_UU = pd.read_csv(os.path.join(dir_base, "user_mentions_user_id_politifact.csv"))
df_UU.to_csv(os.path.join(output_dir, "original_data", "edges", "user_mentions_user.csv"), index=False)
UU_tensor = edges_encoding(df_UU)
UU_rev_tensor = edges_rev_encoding(df_UU)
torch.save(UU_tensor, os.path.join(output_dir, "heterodata", "edgelists", "user_mentions_user.pt"))
torch.save(UU_rev_tensor, os.path.join(output_dir, "heterodata", "edgelists", "user_is_mentioned_by_user.pt"))
'''

""" Applying PCA"""

#dim = 128
'''
NX = torch.load(os.path.join(output_dir, "heterodata", "features", "NX_tensor.pt"))
pca_news = PCA(n_components=dim).fit(NX)
torch.save(torch.tensor(pca_news.transform(NX)).float(), os.path.join(output_dir, "heterodata", "features", f"NX_{dim}_tensor.pt"))
TX = torch.load(os.path.join(output_dir, "heterodata", "features", "TX_tensor_v2.pt"))
pca_tweet = PCA(n_components=dim).fit(TX)
torch.save(torch.tensor(pca_tweet.transform(TX)).float(), os.path.join(output_dir, "heterodata", "features", f"TX_{dim}_tensor_v2.pt"))

UX = torch.load(os.path.join(output_dir, "heterodata", "features", "UX_tensor.pt"))
pca_user = PCA(n_components=dim).fit(UX)
torch.save(torch.tensor(pca_user.transform(UX)).float(), os.path.join(output_dir, "heterodata", "features", f"UX_{dim}_tensor.pt"))

HX = torch.load(os.path.join(output_dir, "heterodata", "features", "HX_tensor.pt"))
pca_hashtag = PCA(n_components=dim).fit(HX)
torch.save(torch.tensor(pca_hashtag.transform(HX)).float(), os.path.join(output_dir, "heterodata", "features", f"HX_{dim}_tensor.pt"))
'''

def load_politifact_heterodata():
    base_dir = "/mnt/nas/martirano/politifact_cleaned/heterodata"
    nodes_dir = os.path.join(base_dir, 'features')
    edges_dir = os.path.join(base_dir, 'edgelists')

    # Load node features

    dim = 128

    NX = torch.load(os.path.join(nodes_dir, 'NX_'+str(dim)+'_tensor.pt'))
    NY = torch.load(os.path.join(base_dir, 'NY_tensor.pt'))
    TX = torch.load(os.path.join(nodes_dir, 'TX_' + str(dim) + '_tensor_v2.pt'))
    UX = torch.load(os.path.join(nodes_dir, 'UX_'+str(dim)+'_tensor.pt'))
    HX = torch.load(os.path.join(nodes_dir, 'HX_'+str(dim)+'_tensor.pt'))

    # Load edgelists

    # tweet_discusses_claim
    TN_tensor = torch.load(os.path.join(edges_dir, 'tweet_discusses_news.pt'))
    NT_tensor = torch.load(os.path.join(edges_dir, 'news_is_discussed_by_tweet.pt'))

    # tweet_has_hashtag_hashtag
    TH_tensor = torch.load(os.path.join(edges_dir, 'tweet_has_hashtag_hashtag.pt'))
    HT_tensor = torch.load(os.path.join(edges_dir, 'hashtag_is_hashtag_of_tweet.pt'))

    # user_posted_tweet
    UT_tensor =torch.load(os.path.join(edges_dir, 'user_posted_tweet.pt'))
    TU_tensor =torch.load(os.path.join(edges_dir, 'tweet_is_posted_by_user.pt'))

    # user_retweeted_tweet
    UR_tensor =torch.load(os.path.join(edges_dir, 'user_retweeted_tweet.pt'))
    RU_tensor =torch.load(os.path.join(edges_dir, 'tweet_is_retweeted_by_user.pt'))


    # user_mentions_user
    UU_tensor =torch.load(os.path.join(edges_dir, 'user_mentions_user.pt'))
    UU_rev_tensor =torch.load(os.path.join(edges_dir, 'user_is_mentioned_by_user.pt'))

    data = HeteroData()

    # NODES

    data['news'].x = NX
    data['news'].y = NY
    data['tweet'].x = TX
    data['user'].x = UX
    data['hashtag'].x = HX

    # EDGES

    data['tweet', 'discusses', 'news'].edge_index = TN_tensor
    data['news', 'is_discussed_by', 'tweet'].edge_index = NT_tensor

    data['tweet', 'has_hashtag', 'hashtag'].edge_index = TH_tensor
    data['hashtag', 'is_hashtag_of', 'tweet'].edge_index = HT_tensor

    data['user', 'posted', 'tweet'].edge_index = UT_tensor
    data['tweet', 'is_posted_by', 'user'].edge_index = TU_tensor

    data['user', 'retweeted', 'tweet'].edge_index = UR_tensor
    data['tweet', 'is_retweeted_by', 'user'].edge_index = RU_tensor


    data['user', 'mentions', 'user'].edge_index = UU_tensor
    data['user', 'is_mentioned_by', 'user'].edge_index = UU_tensor


    # Add metapaths # CTUTC, CTHTC

    metapaths = [
                 [('user', 'retweeted', 'tweet'),
                 ('tweet', 'is_posted_by', 'user')]
                ]

    '''metapaths = [[('claim', 'is_discussed_by', 'tweet'),
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
                      ('tweet', 'discusses', 'claim')]]'''  # CTRTC_q


    data = AddMetaPaths(metapaths, weighted=True)(data)

    transform = T.RandomNodeSplit()
    data = transform(data)

    data = data.to(get_device())

    return data

data = load_politifact_heterodata()
print(data)