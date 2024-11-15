import os
import shutil


dataset_name = "openalex"
n_snapshot = "0"

authors_to_keep = ["A5004271128", "A5087188826", "A5062823200", "A5053183805", "A5034587660", "A5028089542", "A5038199341"] # gli ultimi due sono presenti sia nello 0 che nell'1
#source_base_path = "/Users/francesco/Software/Python/GNN_ContinualLearning/data"
#source_base_path = r"C:\Users\lmart\PycharmProjects\GNN_ContinualLearning\data"
#source_base_path = "/mnt/nas/martirano"
source_base_path = "/home/scala/projects/GNN_ContinualLerning/data"

#####################################################################

if not os.path.exists(f"{source_base_path}/potato_{dataset_name}"):
    shutil.copytree(f"{source_base_path}/{dataset_name}", f"{source_base_path}/potato_{dataset_name}")

# cleaning authors
authors_file_paths = [f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/author_labels.csv",
                      f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/author_is_affiliated_with_institution.csv",
                      f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/author_writes_paper.csv",
                      f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/institution_is_affiliation_of_author.csv",
                      f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/paper_is_written_by_author.csv",
                      f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/nodes/authors.csv"]

for file_path in authors_file_paths:
    with open(file_path, "r") as file:
        first = True
        for line in file:
            if first:
                output_text = line
                first = False

            for word in authors_to_keep:
                if word in line:
                    output_text += line
        output_text += "\n"
    with open(file_path, "w") as file:
        file.write(output_text)

# cleaning papers
authors_writes_file_path = f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/paper_is_written_by_author.csv"
papers_to_keep = []

with open(authors_writes_file_path) as file:
    for line in file:
        for word in authors_to_keep:
            if word in line:
                papers_to_keep.append(line[:line.index(",")])

papers_file_paths = [f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/paper_cites_paper.csv",
                     f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/paper_is_cited_by_paper.csv",
                     f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/nodes/papers.csv"]

for file_path in papers_file_paths:
    with open(file_path, "r") as file:
        first = True
        for line in file:
            if first:
                output_text = line
                first = False

            for word in papers_to_keep:
                if word in line:
                    output_text += line

    with open(file_path, "w") as file:
        file.write(output_text)

# cleaning institutions
authors_affiliation_file_path = f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/edges/institution_is_affiliation_of_author.csv"
affiliations_to_keep = []

with open(authors_affiliation_file_path) as file:
    for line in file:
        for word in authors_to_keep:
            if word in line:
                affiliations_to_keep.append(line[:line.index(",")])

print(affiliations_to_keep)

affiliations_file_paths = [f"{source_base_path}/potato_{dataset_name}/snapshot_{n_snapshot}/original_data/nodes/institutions.csv"]

for file_path in affiliations_file_paths:
    with open(file_path, "r") as file:
        first = True
        for line in file:
            if first:
                output_text = line
                first = False

            for word in affiliations_to_keep:
                if word in line:
                    output_text += line

    with open(file_path, "w") as file:
        file.write(output_text)
