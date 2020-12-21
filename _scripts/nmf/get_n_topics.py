import csv
import os
import re
import sys
sys.path.insert(0, sys.path[0] + "/../..")
import utils.console as console

from evaluation.topic_coherence import coherence_score
from models.nmf import NmfModel
from models.word2vec import Word2VecModel
from preprocessing.preprocess import Preprocessor
from utils.file_helpers import (
    matrix_to_csv, 
    matrix_to_txt, 
    text_to_file, 
    csv_tokens_to_bow, 
    csv_tokens_to_list,
    file_to_list,
)

output_dir = "results/nmf/final"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Preprocess data and obtain BoW
console.info("Getting preprocessed text...")

# Uncomment this for quick loading preprocessed files
dictionary, bow_corpus = csv_tokens_to_bow("results/nmf/final/preprocessed.csv")
corpus = csv_tokens_to_list("results/nmf/final/preprocessed.csv")

console.success("Done!\n")

with open(os.path.join(output_dir, "n_topicsXcoherence.csv"), "w") as score_file:
    writer = csv.writer(score_file)
    writer.writerow(["n_topics", "coherence_score (u_mass)"])

    for n_topics in range(10, 30):

        console.info("Training model...")

        model = NmfModel(
                corpus=bow_corpus, num_topics=n_topics,
                dictionary=dictionary, kappa=0.09)

        console.success("Done!\n")

        # Get top 10 topics
        console.info("Obtaining topic distribution...")

        top_topics = model.get_common_topics(formatted=True)
        topics = [pair[1] for pair in top_topics]

        console.success("Done!\n")

        # Evaluation of obtained topics
        console.info("Obtaining topic coherence...")

        coherence = coherence_score(
                topics=topics, corpus=bow_corpus,
                dictionary=dictionary, coherence='u_mass')

        writer.writerow([str(n_topics), str(coherence)])


        console.success("Done!\n")

