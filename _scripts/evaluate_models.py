import sys
sys.path.insert(0, sys.path[0] + "/..")

import re

from evaluation.topic_coherence import coherence_score
from utils.file_helpers import csv_tokens_to_bow
from utils.console import format_stdout_string

# List of preprocessed texts to be used
preprocessed_corpus = [
    "only_phraser_nohtml",
    "only_phraser_3_gram_nohtml",
    # "only_phraser_4_gram_nohtml",
    "both_merges_nohtml",
]

# Configuration of each model to be used.
# Every entry corresponds to the configuration of
# a individual model.
models_config = [
    {"kappa": 0.4, "minimum_probability": 0.5},
    {"kappa": 0.9, "minimum_probability": 0.1},
]

# Gets coherences
for corpus_file in preprocessed_corpus:
    for configuration in models_config:
        model_name = "kappa={}, " + \
                "minimum_probability={}, " + \
                "data_file={}"
        
        model_name = model_name.format(
                configuration["kappa"], 
                configuration["minimum_probability"], 
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

        