"""
Solo snapshot_0 e snapshot_1
prendo 50 autori random da snapshot_0 e la relativa rete
prendo 30 autori da snapshot_1 t.c. 20 sono nuovi e 10 esistono anche in snapshot_0 (e la relativa rete)
"""

"""
from: data > dataset_name > snapshot_i > original_data (nodes, edges)
to: data > dataset_name > snapshot_i > heterodata (features, edgelists, mapping.pkl, K_old_nodes.pkl, K_new_nodes.pkl,
k_old_edges.pkl, K_new_edges.pkl)

N.B.:
mapping is in the form --> node_type (e.g. author) : {str_id : id}
K_nodes is in the form --> node_type (e.g. author): list of ids
K_edges is in the form --> edge_type (e.g. author_writes_paper): list of pairs (lists) of ids
"""

import os
import shutil
import pandas as pd


from src.data_utils import  get_base_dir


dataset_name = "openalex"
source_base_path = get_base_dir()

if not os.path.exists(f"{source_base_path}/{dataset_name}_subset"):
    shutil.copytree(f"{source_base_path}/{dataset_name}", f"{source_base_path}/{dataset_name}_subset")


original_dir_0 = os.path.join(source_base_path, dataset_name, 'snapshot_0', 'original_data')
original_dir_1 = os.path.join(source_base_path, dataset_name, 'snapshot_1', 'original_data')
subset_dir_0 = os.path.join(source_base_path, f"{dataset_name}_subset", 'snapshot_0', 'original_data')
subset_dir_1 = os.path.join(source_base_path, f"{dataset_name}_subset", 'snapshot_1', 'original_data')

# subset degli autori
autori0 = pd.read_csv(os.path.join(original_dir_0, 'nodes', 'authors.csv'))
autori1 = pd.read_csv(os.path.join(original_dir_1, 'nodes', 'authors.csv'))

autori_comuni = set(autori0['id']).intersection(autori1['id'])
autori_comuni_selezionati = list(autori_comuni)[:10]
subset_comuni_0 = autori0[autori0['id'].isin(autori_comuni_selezionati)]
subset_comuni_1 = autori1[autori1['id'].isin(autori_comuni_selezionati)]

autori0_rimanenti = autori0[~autori0['id'].isin(autori_comuni_selezionati)]
autori1_rimanenti = autori1[~autori1['id'].isin(autori_comuni_selezionati)]

subset_random_0 = autori0_rimanenti.sample(n=40, random_state=42)
id_selezionati_0 = set(subset_comuni_0['id']).union(set(subset_random_0['id']))
subset_random_1 = autori1_rimanenti[~autori1_rimanenti['id'].isin(id_selezionati_0)].sample(n=20, random_state=42) #escludendo i comuni e i 50 selezionati dal primo dataset

autori0_subset_finale = pd.concat([subset_comuni_0, subset_random_0])
autori1_subset_finale = pd.concat([subset_comuni_1, subset_random_1])

autori0_subset_finale.to_csv(os.path.join(subset_dir_0, 'nodes', 'authors.csv'), index=False)
autori1_subset_finale.to_csv(os.path.join(subset_dir_1, 'nodes', 'authors.csv'), index=False)

author_ids_0_sub = pd.read_csv(os.path.join(subset_dir_0, 'nodes', 'authors.csv'))["id"].tolist()
author_ids_1_sub = pd.read_csv(os.path.join(subset_dir_1, 'nodes', 'authors.csv'))["id"].tolist()

author_labels_0 = pd.read_csv(os.path.join(subset_dir_0, 'author_labels.csv'))
author_labels_1 = pd.read_csv(os.path.join(subset_dir_1, 'author_labels.csv'))

author_labels_0_subset = author_labels_0[author_labels_0['id'].isin(author_ids_0_sub)]
author_labels_1_subset = author_labels_1[author_labels_1['id'].isin(author_ids_1_sub)]

author_labels_0_subset.to_csv(os.path.join(subset_dir_0, 'author_labels.csv'), index=False)
author_labels_1_subset.to_csv(os.path.join(subset_dir_1, 'author_labels.csv'), index=False)

print("autori ok")

# subset AI
AI_0 = pd.read_csv(os.path.join(original_dir_0, 'edges', 'author_is_affiliated_with_institution.csv'))
AI_1 = pd.read_csv(os.path.join(original_dir_1, 'edges', 'author_is_affiliated_with_institution.csv'))

AI_0_subset = AI_0[AI_0['src'].isin(author_ids_0_sub)]
AI_1_subset = AI_1[AI_1['src'].isin(author_ids_1_sub)]

AI_0_subset.to_csv(os.path.join(subset_dir_0, 'edges', 'author_is_affiliated_with_institution.csv'), index=False)
AI_1_subset.to_csv(os.path.join(subset_dir_1, 'edges', 'author_is_affiliated_with_institution.csv'), index=False)

print("AI ok")

# subset IA
IA_0 = pd.read_csv(os.path.join(original_dir_0, 'edges', 'institution_is_affiliation_of_author.csv'))
IA_1 = pd.read_csv(os.path.join(original_dir_1, 'edges', 'institution_is_affiliation_of_author.csv'))

IA_0_subset = IA_0[IA_0['tgt'].isin(author_ids_0_sub)]
IA_1_subset = IA_1[IA_1['tgt'].isin(author_ids_1_sub)]

IA_0_subset.to_csv(os.path.join(subset_dir_0, 'edges', 'institution_is_affiliation_of_author.csv'), index=False)
IA_1_subset.to_csv(os.path.join(subset_dir_1, 'edges', 'institution_is_affiliation_of_author.csv'), index=False)

print("IA ok")

# subset delle institutions
institutions_0 = pd.read_csv(os.path.join(original_dir_0, 'nodes', 'institutions.csv'))
institutions_1 = pd.read_csv(os.path.join(original_dir_1, 'nodes', 'institutions.csv'))

institutions_ids_0_sub = pd.read_csv(os.path.join(subset_dir_0, 'edges', 'author_is_affiliated_with_institution.csv'))["tgt"].tolist()
institutions_ids_1_sub = pd.read_csv(os.path.join(subset_dir_1, 'edges', 'author_is_affiliated_with_institution.csv'))["tgt"].tolist()

institutions_0_subset = institutions_0[institutions_0['id'].isin(institutions_ids_0_sub)]
institutions_1_subset = institutions_1[institutions_1['id'].isin(institutions_ids_1_sub)]

institutions_0_subset.to_csv(os.path.join(subset_dir_0, 'nodes', 'institutions.csv'), index=False)
institutions_1_subset.to_csv(os.path.join(subset_dir_1, 'nodes', 'institutions.csv'), index=False)

print("istituzioni ok")

# subset AP
AP_0 = pd.read_csv(os.path.join(original_dir_0, 'edges', 'author_writes_paper.csv'))
AP_1 = pd.read_csv(os.path.join(original_dir_1, 'edges', 'author_writes_paper.csv'))

AP_0_subset = AP_0[AP_0['src'].isin(author_ids_0_sub)]
AP_1_subset = AP_1[AP_1['src'].isin(author_ids_1_sub)]

AP_0_subset.to_csv(os.path.join(subset_dir_0, 'edges', 'author_writes_paper.csv'), index=False)
AP_1_subset.to_csv(os.path.join(subset_dir_1, 'edges', 'author_writes_paper.csv'), index=False)

print("AP ok")

# subset PA
PA_0 = pd.read_csv(os.path.join(original_dir_0, 'edges', 'paper_is_written_by_author.csv'))
PA_1 = pd.read_csv(os.path.join(original_dir_1, 'edges', 'paper_is_written_by_author.csv'))

PA_0_subset = PA_0[PA_0['tgt'].isin(author_ids_0_sub)]
PA_1_subset = PA_1[PA_1['tgt'].isin(author_ids_1_sub)]

PA_0_subset.to_csv(os.path.join(subset_dir_0, 'edges', 'paper_is_written_by_author.csv'), index=False)
PA_1_subset.to_csv(os.path.join(subset_dir_1, 'edges', 'paper_is_written_by_author.csv'), index=False)

print("PA ok")

# subset dei papers
papers_0 = pd.read_csv(os.path.join(original_dir_0, 'nodes', 'papers.csv'))
papers_1 = pd.read_csv(os.path.join(original_dir_1, 'nodes', 'papers.csv'))

papers_ids_0_sub = pd.read_csv(os.path.join(subset_dir_0, 'edges', 'author_writes_paper.csv'))["tgt"].tolist()
papers_ids_1_sub = pd.read_csv(os.path.join(subset_dir_1, 'edges', 'author_writes_paper.csv'))["tgt"].tolist()

papers_0_subset = papers_0[papers_0['id'].isin(papers_ids_0_sub)]
papers_1_subset = papers_1[papers_1['id'].isin(papers_ids_1_sub)]

papers_0_subset.to_csv(os.path.join(subset_dir_0, 'nodes', 'papers.csv'), index=False)
papers_1_subset.to_csv(os.path.join(subset_dir_1, 'nodes', 'papers.csv'), index=False)

print("paper ok")

# subset PP
PP_0 = pd.read_csv(os.path.join(original_dir_0, 'edges', 'paper_cites_paper.csv'))
PP_0_rev = pd.read_csv(os.path.join(original_dir_0, 'edges', 'paper_is_cited_by_paper.csv'))
PP_1 = pd.read_csv(os.path.join(original_dir_1, 'edges', 'paper_cites_paper.csv'))
PP_1_rev = pd.read_csv(os.path.join(original_dir_1, 'edges', 'paper_is_cited_by_paper.csv'))

PP_0_subset = PP_0[(PP_0['src'].isin(papers_ids_0_sub)) & (PP_0['tgt'].isin(papers_ids_0_sub))]
PP_0_rev_subset = PP_0_rev[(PP_0_rev['src'].isin(papers_ids_0_sub)) & (PP_0_rev['tgt'].isin(papers_ids_0_sub))]
PP_1_subset = PP_1[(PP_1['src'].isin(papers_ids_1_sub)) & (PP_1['tgt'].isin(papers_ids_1_sub))]
PP_1_rev_subset = PP_1_rev[(PP_1_rev['src'].isin(papers_ids_1_sub)) & (PP_1_rev['tgt'].isin(papers_ids_1_sub))]

PP_0_subset.to_csv(os.path.join(subset_dir_0, 'edges', 'paper_cites_paper.csv'), index=False)
PP_0_rev_subset.to_csv(os.path.join(subset_dir_0, 'edges', 'paper_is_cited_by_paper.csv'), index=False)
PP_1_subset.to_csv(os.path.join(subset_dir_1, 'edges', 'paper_cites_paper.csv'), index=False)
PP_1_rev_subset.to_csv(os.path.join(subset_dir_1, 'edges', 'paper_is_cited_by_paper.csv'), index=False)

print("PP ok")

