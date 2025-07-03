"""
from: miscellaneous
to: data > dataset_name > snapshot_i > original_data (nodes, edges)
"""
import os
import pandas as pd
import json
import pickle
import torch

from data.data_utils import save_dict_to_pickle, open_pickle
from data.graph_loader import build_heterodata
#from utils import get_base_dir

dataset_name = "openalex"
base_dir = f"/home/martirano/data/{dataset_name}"
or_dir = os.path.join(base_dir, "raw/original_data")



def split_original_data():

    # PAPERS
    print("##### Processing PAPERS #####")
    papers = pd.read_csv(os.path.join(or_dir, 'papers.csv'))
    papers_abs = pd.read_csv(os.path.join(or_dir, 'paper_abstracts.csv'), delimiter=';') #id, abstract
    papers_con = pd.read_csv(os.path.join(or_dir, 'paper_concepts.csv')) ##paper_id, concept_id, concept_name, level
    papers_con.rename(columns={'paper_id': 'id'}, inplace=True)
    papers_con_ok = papers_con.groupby('id').apply(lambda x: list(zip(x['concept_id'], x['concept_name'], x['level']))).reset_index(name='concepts')

    years = sorted(papers['publication_year'].drop_duplicates().tolist())
    years = years[:-1]

    for i,year in enumerate(years):
        print(year)

        #create directory structure
        full_path_nodes = os.path.join(base_dir, f'snapshot_{i}', 'original_data/nodes')
        os.makedirs(full_path_nodes, exist_ok=True)
        full_path_edges = os.path.join(base_dir, f'snapshot_{i}', 'original_data/edges')
        os.makedirs(full_path_edges, exist_ok=True)

        papers_sub = papers.loc[papers['publication_year'] == year]
        papers_sub = papers_sub.sort_values(by='id', ascending=True).reset_index()
        papers_sub.drop(columns='index', inplace=True)
        papers_ok = papers_sub.id.tolist()
        print(f"Papers ok: {papers_sub.shape}")
        papers_abs_sub = papers_abs[papers_abs['id'].isin(papers_ok)]
        print(f"Paper-abstracts {papers_abs_sub.shape}")
        papers_con_sub = papers_con_ok[papers_con_ok['id'].isin(papers_ok)]
        print(f"Paper-concepts {papers_con_sub.shape}")
        papers_meta_sub = papers_abs_sub.merge(papers_con_sub, on='id', how='left')
        papers_meta_sub = papers_meta_sub.sort_values(by='id', ascending=True).reset_index()
        papers_meta_sub.drop(columns='index', inplace=True)
        papers_meta_sub.rename(columns={'id': 'id2'}, inplace=True)
        print(f"Paper-metadata {papers_meta_sub.shape}")
        #dfP = papers_sub.merge(papers_meta_sub, on='id')
        dfP = pd.concat([papers_sub, papers_meta_sub], axis=1)
        #mismatched_count = len(dfP[dfP['id'] != dfP['id2']])
        #print(f"Mismatched papers: {mismatched_count}")
        dfP.drop(columns='id2', inplace=True)
        print(f"Paper-all {dfP.shape}")
        dfP.to_csv(os.path.join(full_path_nodes, 'paper.csv'), index=False)

    # PP
    print("##### Processing PP #####")
    PP = pd.read_csv(os.path.join(or_dir, 'PP.csv')) #citing, cited, year
    years = sorted(PP['year'].drop_duplicates().tolist())
    years = years[:-1]

    for i,year in enumerate(years):
        print(year)
        PP_sub = PP.loc[PP['year'] == year]
        print(f"PP {PP_sub.shape}")

        papers_all = pd.read_csv(os.path.join(or_dir, 'papers.csv'))
        papers_prev = papers_all.loc[papers_all['publication_year'] <= year]
        papers_prev_ids = papers_prev.id.tolist()

        PP_sub_ok = PP_sub[PP_sub["cited"].isin(papers_prev_ids)]

        paper_cites_paper = PP_sub_ok.copy()
        paper_cites_paper.rename(columns={'citing': 'src', 'cited':'tgt'}, inplace=True)
        paper_cites_paper.to_csv(os.path.join(base_dir, f'snapshot_{i}', 'original_data/edges/paper_cites_paper.csv'),index=False)

        #check no citing of other years
        #P_year = pd.read_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/nodes/papers.csv'))
        #papers_year = P_year['id'].tolist()
        #PP_sub_ok = PP_sub[PP_sub['citing'].isin(papers_year)]
        #print(f"PP ok {PP_sub_ok.shape}")

        paper_is_cited_by_paper = PP_sub_ok[['cited', 'citing', 'year']].copy()
        paper_is_cited_by_paper.rename(columns={'cited': 'src', 'citing': 'tgt'}, inplace=True)
        paper_is_cited_by_paper.to_csv(os.path.join(base_dir, f'snapshot_{i}', 'original_data/edges/paper_is_cited_by_paper.csv'), index=False)


    # AP
    print("##### Processing AP #####")
    AP = pd.read_csv(os.path.join(or_dir, 'PA.csv')) #paper, #author, #position

    for i,year in enumerate(years):
        print(year)
        P_year = pd.read_csv(os.path.join(base_dir, f'snapshot_{i}', 'original_data/nodes/paper.csv'))
        papers_year = P_year['id'].tolist()
        AP_sub = AP[AP['paper'].isin(papers_year)]
        print(f"AP {AP_sub.shape}")

        paper_is_written_by_author = AP_sub.copy()
        paper_is_written_by_author.rename(columns={'paper': 'src', 'author': 'tgt'}, inplace=True)
        paper_is_written_by_author.to_csv(os.path.join(base_dir, f'snapshot_{i}', 'original_data/edges/paper_is_written_by_author.csv'), index=False)

        author_writes_paper = AP_sub[['author', 'paper', 'position']].copy()
        author_writes_paper.rename(columns={'author': 'src', 'paper': 'tgt'}, inplace=True)
        author_writes_paper.to_csv(os.path.join(base_dir, f'snapshot_{i}','original_data/edges/author_writes_paper.csv'), index=False)


    # AUTHORS
    print("##### Processing AUTHORS #####")
    authors_metadata = pd.read_csv(os.path.join(or_dir, 'authors_metadata.csv'))
    print(f"authors metadata {authors_metadata.shape}")

    for i,year in enumerate(years):
        print(year)
        AP = pd.read_csv(os.path.join(base_dir, f'snapshot_{i}','original_data/edges/author_writes_paper.csv'))
        A_year = AP['src'].drop_duplicates().tolist()
        authors_metadata_sub = authors_metadata[authors_metadata['id'].isin(A_year)]
        print(f"authors metadata {authors_metadata_sub.shape}")
        authors_metadata_sub.to_csv(os.path.join(base_dir, f'snapshot_{i}', 'original_data/nodes/author.csv'), index=False)

        #labels
        filename = os.path.join(base_dir, f'snapshot_{i}', 'original_data/nodes/author.csv')
        df = pd.read_csv(filename, usecols=["id", "x_concepts"])
        df_sub = df[df["id"].isin(A_year)]
        df_new = labels_generation(df_sub, A_year)
        print(f"authors labels {df_new.shape[0]}")
        df_new.to_csv(
            os.path.join(base_dir, f'snapshot_{i}', 'original_data/author_labels.csv'),
            index=False)

    #AI
    print("##### Processing AI #####")
    AI = pd.read_csv(os.path.join(or_dir, 'AI.csv')) #author_id, institution_id
    print(f"authors institutions {AI.shape}")

    for i,year in enumerate(years):
        print(year)
        authors = pd.read_csv(os.path.join(base_dir, 'snapshot_' + str(i),'original_data/nodes/author.csv'))
        A_year = authors['id'].drop_duplicates().tolist()
        AI_sub = AI[AI['author_id'].isin(A_year)]
        print(f"AI {AI_sub.shape}")

        author_is_affiliated_with_institution = AI_sub.copy()
        author_is_affiliated_with_institution.rename(columns={'author_id': 'src', 'institution_id': 'tgt'}, inplace=True)
        author_is_affiliated_with_institution.to_csv(os.path.join(base_dir, 'snapshot_' + str(i), 'original_data/edges/author_is_affiliated_with_institution.csv'), index=False)

        institution_is_affiliation_of_author = AI_sub[['institution_id', 'author_id']].copy()
        institution_is_affiliation_of_author.rename(columns={'institution_id': 'src', 'author_id': 'tgt'}, inplace=True)
        institution_is_affiliation_of_author.to_csv(os.path.join(base_dir, 'snapshot_' + str(i), 'original_data/edges/institution_is_affiliation_of_author.csv'), index=False)

    #INSTITUTIONS
    print("##### Processing INSTITUTIONS #####")
    institutions = pd.read_csv(os.path.join(or_dir, 'institutions.csv'))
    print(f"institutions {institutions.shape}")

    for i,year in enumerate(years):
        print(year)
        AI = pd.read_csv(os.path.join(base_dir, 'snapshot_' + str(i),'original_data/edges/author_is_affiliated_with_institution.csv'))
        I_year = AI['tgt'].drop_duplicates().tolist()
        institutions_sub = institutions[institutions['id'].isin(I_year)]
        print(f"institutions {institutions_sub.shape}")
        institutions_sub.to_csv(os.path.join(base_dir, 'snapshot_' + str(i), 'original_data/nodes/institution.csv'), index=False)



def get_sub_concepts_list():
    return ["multimedia", "database", "internet privacy", "natural language processing", "data science",
                    "artificial intelligence", "distributed computing", "computer hardware",
                    "theoretical computer science", "library science", "operating system", "world wide web",
                    "parallel computing", "information retrieval", "computer security", "knowledge management",
                    "computer vision", "data mining", "speech recognition", "programming language",
                    "computer network", "machine learning"] #"computer architecture", "real time computing", "computer graphics images", "human computer interaction",


def labels_generation(df, all_authors):
    #main_concept = "Computer science"
    sub_concepts = get_sub_concepts_list()
    sub_concepts.append("generic")  # no level 1
    authors = []
    labels = []

    for _, row in df.iterrows():
        concepts = json.loads(
            row["x_concepts"].replace("': '", "\": \"").replace("', '", "\", \"").replace("{'", "{\"").replace("'}","\"}").replace("': ", "\": ").replace(", '", ", \"")
        )
        #if contains_main_concept(main_concept, concepts): ###ok always True
        max_score = -1
        max_value = "generic"
        for concept in concepts:
            if concept["display_name"].lower() in sub_concepts:
                if concept["score"] > max_score:
                    max_score = concept["score"]
                    max_value = concept["display_name"].lower()

        authors.append(row["id"])
        labels.append(max_value)

    if len(all_authors) > len(authors):
        missing_authors = [a for a in all_authors if a not in authors]
        for a in missing_authors:
            authors.append(a)
            labels.append("generic")

    df_new = pd.DataFrame({'id': authors, 'label': labels})
    return df_new



#OLD
def create_mapping_labels():
    sub_concepts_ok = ["multimedia", "database", "internet privacy", "natural language processing", "data science",
                    "artificial intelligence", "distributed computing", "computer hardware",
                    "theoretical computer science", "library science", "operating system", "world wide web",
                    "parallel computing", "information retrieval", "computer security", "knowledge management",
                    "computer vision", "data mining", "speech recognition", "programming language",
                    "computer network", "machine learning"] #"computer architecture", "real time computing", "computer graphics images", "human computer interaction",
    labels_dict = {label: index for index, label in enumerate(sub_concepts_ok)}
    with open("/mnt/nas/martirano/openalex/mapping_labels.pkl", "wb") as f:
        pickle.dump(labels_dict, f)


def processing_paper_concepts():
    for i in range(7):
        filename = os.path.join(base_dir, 'snapshot_' + str(i), 'original_data/nodes/paper.csv')
        df = pd.read_csv(filename)
        df['filtered_concepts'] = df['concepts'].apply(filter_concepts)
        df.to_csv(
            os.path.join(base_dir, 'snapshot_' + str(i), 'original_data/nodes/paper.csv'),
            index=False)



def filter_concepts(concepts):
    # Extract the names of the concepts from each tuple, convert them to lowercase,
    # and keep only those that are in sub_concepts_ok_set
    sub_concepts_ok = ["multimedia", "database", "internet privacy", "natural language processing", "data science",
                       "artificial intelligence", "distributed computing", "computer hardware",
                       "theoretical computer science", "library science", "operating system", "world wide web",
                       "parallel computing", "information retrieval", "computer security", "knowledge management",
                       "computer vision", "data mining", "speech recognition", "programming language",
                       "computer network", "machine learning"]
    filtered = [concept[1].lower() for concept in eval(concepts) if concept[1].lower() in sub_concepts_ok]
    return filtered




def get_new_knowledge(dataset_name, snapshot):
    new_nodes = {}
    old_nodes = {}
    current_graph = build_heterodata(dataset_name, snapshot)
    for ntype in current_graph.node_types:
        old_nodes[ntype] = []
        new_nodes[ntype] = []

    if snapshot == 0:
        # in new vanno tutti i nuovi
        for ntype in current_graph.node_types:
            num_new = current_graph[ntype].x.shape[0]
            new_nodes[ntype] = list(range(num_new))

    else:
        old_graph = build_heterodata(dataset_name, snapshot-1)
        # in new vanno intanto gli unseen
        for ntype in current_graph.node_types:
            num_old = old_graph[ntype].x.shape[0]
            num_new = current_graph[ntype].x.shape[0]
            new_nodes[ntype] = list(range(num_old, num_new))
        # aggiungiamo i changed a new e gli unchanged ad old
        for etype in current_graph.edge_types:
            ntype_src = etype[0]
            ntype_tgt = etype[2]
            for edge in old_graph.edge_index_dict[etype]:
                if edge[0].item() in new_nodes[ntype_src]:
                    new_nodes[ntype_tgt].append(edge[1].item())
                else:
                    old_nodes[ntype_tgt].append(edge[1].item())
                if edge[1].item() in new_nodes[ntype_tgt]:
                    new_nodes[ntype_src].append(edge[0].item())
                else:
                    old_nodes[ntype_src].append(edge[0].item())

        for k in new_nodes:
            new_nodes[k] = list(set(new_nodes[k]))
        for k in old_nodes:
            old_nodes[k] = list(set(old_nodes[k]))

    return new_nodes, old_nodes




if __name__ == "__main__":

    # split_original_data() #STEP 1
    # processing_paper_concepts() #STEP2
    for i in range(7): #STEP 4
        get_new_knowledge(dataset_name, i)