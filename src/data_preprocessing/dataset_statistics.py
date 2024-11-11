import json
import pandas


filename = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/original_data/nodes/authors.csv"
main_concept = "Computer science"
sub_concepts = ["multimedia", "database", "internet privacy", "natural language processing", "data science", "artificial intelligence", "real time computing", "distributed computing", "computer hardware", "theoretical computer science", "computer architecture", "computer graphics images", "library science", "operating system", "world wide web", "parallel computing", "information retrieval", "computer security", "knowledge management", "computer vision", "data mining", "speech recognition", "programming language", "human computer interaction", "computer network", "machine learning"]


######################################################################


def contains_main_concept(main_concept, concepts):
    for i in range(len(concepts)):
        if main_concept.lower() in concepts[i]["display_name"].lower():
            return True
    
    return False


if __name__ == "__main__":
    n_main_concept = 0
    n_not_main_concept = 0
    df = pandas.read_csv(filename, usecols=["x_concepts"])
    sub_concepts.append("none")
    concepts_occurrences = dict.fromkeys(sub_concepts, 0)
    for _, row in df.iterrows():
        concepts = json.loads(row["x_concepts"].replace("': '", "\": \"").replace("', '", "\", \"").replace("{'", "{\"").replace("'}", "\"}").replace("': ", "\": ").replace(", '", ", \""))
        if contains_main_concept(main_concept, concepts):
            n_main_concept += 1
            max_score = -1
            max_value = "none"
            for concept in concepts:
                if concept["display_name"].lower() in sub_concepts:
                    if concept["score"] > max_score:
                        max_score = concept["score"]
                        max_value = concept["display_name"].lower()

            concepts_occurrences[max_value] += 1

        else:
            n_not_main_concept += 1

    print(f"There are {n_main_concept} authors that contains the main concept '{main_concept}' and {n_not_main_concept} authors that do not contain it.")
    print(f"Detailed count: ")
    sorted_values = sorted(concepts_occurrences.items(), key=lambda x: x[1], reverse=True)
    for i in sorted_values:
        print(f"\t- {i[0]}: {i[1]}")
