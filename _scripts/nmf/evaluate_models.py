import csv
import os
import re
import sys
sys.path.insert(0, sys.path[0] + "/../..")

from evaluation.topic_coherence import coherence_score
from utils.file_helpers import csv_tokens_to_bow
from utils.console import format_stdout_string

preprocessed_corpus = [
    "only_phraser",
    "only_phraser_with_digits",
    "both_merges",
    "both_merges_with_digits",
]

parameters = {
    "kappa": [0.09, 0.4, 0.9],
    "minimum_probability": [0.5, 1.5],
}


# Gets coherences
with open("results/nmf/models_coherence.csv", "w") as coherence_file:
    writer = csv.writer(coherence_file)
    writer.writerow(["kappa", "minimum_probability", "data_file", "coherence_score"])

    for corpus_file in preprocessed_corpus:
        for kappa in parameters["kappa"]:
            for minimum_probability in parameters["minimum_probability"]:
                model_name = "kappa={}, " + \
                        "minimum_probability={}, " + \
                        "data_file={}"
            
                model_name = model_name.format(
                        kappa, 
                        minimum_probability, 
                        corpus_file)

                dictionary, corpus = csv_tokens_to_bow(
                        "data/preprocessed/" + corpus_file + ".csv")

                topics = []
                with open("results/nmf/topics/" + model_name, "r") as topic_file:
                    for line in topic_file:
                        topic_words = re.findall("\"[^\"]+\"", line)
                        topics.append([word[1:-1] for word in topic_words])
                
                coherence = coherence_score(
                        topics=topics, corpus=corpus, 
                        dictionary=dictionary, coherence='u_mass')

                print(format_stdout_string(
                        "Model:", color="red", 
                        bright=True), model_name)
                print(format_stdout_string(
                        "\tScore", color="red", bold=True),
                        coherence)

                writer.writerow([
                    kappa,
                    minimum_probability,
                    corpus_file,
                    coherence,
                ])

        