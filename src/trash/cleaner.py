import os
import shutil


dataset_name = "openalex"
n_snapshot = "1"

authors_to_keep = ["A5057792892", "A5020474912", "A5084966916", "A5003122432", "A5047160693", "A5030402987", "A5078277073", "A5017974535", "A5047487276", "A5041151162", "A5026788167", "A5055305609", "A5018151377", "A5039494431", "A5033579705", "A5041520103", "A5029032085", "A5007140671", "A5086526106", "A5026286215", "A5043069049", "A5052608804", "A5047059242", "A5077461494", "A5008888706", "A5013800136", "A5055313324", "A5067970454", "A5058138023", "A5006543237", "A5005605270", "A5071241846", "A5010462308", "A5019257524", "A5000763047", "A5015563134", "A5036631153", "A5028078726", "A5027544704", "A5090641466", "A5074481639", "A5028729642", "A5069267601", "A5072816665", "A5083209573", "A5003587558", "A5059799200", "A5005891715", "A5029477698", "A5090732704", "A5039945857", "A5061387861", "A5051819150", "A5015785795", "A5049922472", "A5026962283", "A5008509744", "A5066302861", "A5046340343", "A5059436198", "A5049560556", "A5045499382", "A5006380777", "A5015753842", "A5058360953", "A5013114471", "A5016864296", "A5060840266", "A5067007738", "A5057804144", "A5005566691", "A5072892151", "A5032651196", "A5045895774", "A5016338914", "A5012632345", "A5007002585", "A5024835217", "A5021926457", "A5034143246", "A5047359547", "A5056390625", "A5007087783", "A5068502672", "A5087152917", "A5031760215", "A5039939708", "A5009470104", "A5073031308", "A5036067425", "A5037446815", "A5082367164", "A5066915624", "A5069839542", "A5074143529", "A5052396894", "A5020390883", "A5080867114", "A5004271128", "A5087188826", "A5062823200", "A5053183805", "A5034587660", "A5028089542", "A5038199341"] # gli ultimi due sono presenti sia nello 0 che nell'1
#source_base_path = "/Users/francesco/Software/Python/GNN_ContinualLearning/data"
#source_base_path = r"C:\Users\lmart\PycharmProjects\GNN_ContinualLearning\data"
source_base_path = "/mnt/nas/martirano"
#source_base_path = "/home/scala/projects/GNN_ContinualLerning/data"

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
