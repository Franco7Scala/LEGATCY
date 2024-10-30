"""
from: miscellaneous
to: data > dataset_name > snapshot_i > original_data (nodes, edges)
"""
import os
import pandas as pd


dataset_name = "openalex"
or_dir = '/mnt/nas/martirano/openalex/raw/original_data'

# PAPERS
"""
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
    full_path_nodes = os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/nodes')
    os.makedirs(full_path_nodes, exist_ok=True)
    full_path_edges = os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/edges')
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
    dfP.to_csv(os.path.join(full_path_nodes, 'papers.csv'), index=False)
"""

# PP
PP = pd.read_csv(os.path.join(or_dir, 'PP.csv')) #citing, cited, year
years = sorted(PP['year'].drop_duplicates().tolist())
years = years[:-1]

"""
for i,year in enumerate(years):
    print(year)
    PP_sub = PP.loc[PP['year'] == year]
    print(f"PP {PP_sub.shape}")

    paper_cites_paper = PP_sub.copy()
    paper_cites_paper.rename(columns={'citing': 'src', 'cited':'tgt'}, inplace=True)
    paper_cites_paper.to_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/edges/paper_cites_paper.csv'),index=False)

    #check no citing of other years
    #P_year = pd.read_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/nodes/papers.csv'))
    #papers_year = P_year['id'].tolist()
    #PP_sub_ok = PP_sub[PP_sub['citing'].isin(papers_year)]
    #print(f"PP ok {PP_sub_ok.shape}")

    paper_is_cited_by_paper = PP_sub[['cited', 'citing', 'year']].copy()
    paper_is_cited_by_paper.rename(columns={'cited': 'src', 'citing': 'tgt'}, inplace=True)
    paper_is_cited_by_paper.to_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/edges/paper_is_cited_by_paper.csv'), index=False)
 """

# AP
"""
AP = pd.read_csv(os.path.join(or_dir, 'PA.csv')) #paper, #author, #position

for i,year in enumerate(years):
    print(year)
    P_year = pd.read_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/nodes/papers.csv'))
    papers_year = P_year['id'].tolist()
    AP_sub = AP[AP['paper'].isin(papers_year)]
    print(f"AP {AP_sub.shape}")

    paper_is_written_by_author = AP_sub.copy()
    paper_is_written_by_author.rename(columns={'paper': 'src', 'author': 'tgt'}, inplace=True)
    paper_is_written_by_author.to_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i), 'original_data/edges/paper_is_written_by_author.csv'), index=False)

    author_writes_paper = AP_sub[['author', 'paper', 'position']].copy()
    author_writes_paper.rename(columns={'author': 'src', 'paper': 'tgt'}, inplace=True)
    author_writes_paper.to_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i),'original_data/edges/author_writes_paper.csv'), index=False)
"""

# AUTHORS
authors_metadata = pd.read_csv(os.path.join(or_dir, 'authors_metadata.csv'))
print(f"authors metadata {authors_metadata.shape}")
print(authors_metadata.head(5))

"""
for i,year in enumerate(years):
    print(year)
    AP = pd.read_csv(os.path.join('/mnt/nas/martirano/openalex', 'snapshot_' + str(i),'original_data/edges/author_writes_paper.csv'))
    A_year = AP['src'].drop_duplicates().tolist()
"""

#AI

#INSTITUTIONS






