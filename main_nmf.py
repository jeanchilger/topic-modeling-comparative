"""
Generate preprocessed files, model and topics for
NMF model.

Execute:
    python3 main_nmf.py
"""

import os
import re
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
)

dataset_path = "data/news.csv"
output_dir = "results/nmf/final"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Preprocess data and obtain BoW
console.info("Preprocessing text...")
preprocessed = Preprocessor(
        dataset_path,
        columns=["title", "description", "text"])

corpus = preprocessed.corpus
bow_corpus = preprocessed.bag_of_words
dictionary = preprocessed.dictionary

matrix_to_csv(
        preprocessed.corpus,
        os.path.join(output_dir, "preprocessed.csv"))

matrix_to_csv(
        preprocessed.corpus,
        os.path.join(output_dir, "preprocessed.txt"))

console.success("Done!\n")

# Uncomment this for quick loading preprocessed files
# dictionary, bow_corpus = csv_tokens_to_bow("results/nmf/final/preprocessed.csv")
# corpus = csv_tokens_to_list("results/nmf/final/preprocessed.csv")

# Train Model
console.info("Training model...")

model = NmfModel(
        corpus=bow_corpus, num_topics=25,
        dictionary=dictionary, kappa=0.4)

model.save(os.path.join(output_dir, "model"))

# Uncomment this for quick loading pre-trained model
# model = NmfModel(model_name="results/nmf/final/model", corpus=bow_corpus)

console.success("Done!\n")

# Get top 10 topics
console.info("Obtaining topic distribution...")

topics = []
top_topics = model.get_common_topics()
for topic_dist in top_topics:
    topics.append([pair[0] for pair in topic_dist])

matrix_to_txt(
        topics, 
        os.path.join(output_dir, 
        "topics.txt"), join_columns=True)

console.success("Done!\n")

# Evaluation of obtained topics
console.info("Obtaining topic coherence...")

coherence = coherence_score(
        topics=topics, corpus=bow_corpus,
        dictionary=dictionary, coherence='u_mass')

text_to_file(str(coherence), os.path.join(output_dir, "evaluation.txt"))

console.success("Done!\n")

# Getting words similar to topic words
console.info("Get most similar words to topic words...")

word2vec_model = Word2VecModel(
        sentences=corpus, vector_size=200, workers=4)

most_similar_output = ""
for topic in topics:
    topic_words = "\n".join(
            ["\t" + str(word) for word 
            in word2vec_model.most_similar_to(topic)])

    console.error(str(topic))
    console.warning(topic_words)
    print()

    most_similar_output += str(topic) + "\n"
    most_similar_output += topic_words + "\n"

    text_to_file(
            most_similar_output, 
            "results/nmf/final/most_similar_to_topics.txt")


console.success("Done!\n")
