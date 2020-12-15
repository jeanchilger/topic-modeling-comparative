import os
import re
import utils.console as console

from evaluation.topic_coherence import coherence_score
from models.nmf import NmfModel
from preprocessing.preprocess import Preprocessor
from utils.file_helpers import matrix_to_csv, matrix_to_txt, text_to_file

dataset_path = "data/news.csv"
output_dir = "results/nmf/final"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# Preprocess data and obtain BoW
# data/preprocessed/only_phraser_with_digits_nohtml.csv
console.info("Preprocessing text...")
preprocessed = Preprocessor(
        dataset_path,
        columns=["title", "description", "text"],
        keep_digits=True)

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


# Train Model
# trained_models/kappa=0.4, minimum_probability=0.1, data_file=only_phraser_with_digits_nohtml
console.info("Training model...")

model = NmfModel(
        corpus=bow_corpus, num_topics=25,
        dictionary=dictionary, kappa=0.4)

model.save(os.path.join(output_dir, "model"))

console.success("Done!\n")

# Get top 10 topics
console.info("Obtaining topic distribution...")

topics = []
top_topics = model.model.show_topics()
for topic_dist in top_topics:
    topic_words = re.findall("\"[^\"]+\"", topic_dist[1])
    topics.append([word[1:-1] for word in topic_words])

top_topics = [topic_dist[1] for topic_dist in top_topics]

matrix_to_txt(top_topics, os.path.join(output_dir, "topics.txt"))

console.success("Done!\n")

# Evaluation of obtained topics
console.info("Obtaining topic coherence...")

coherence = coherence_score(
        topics=topics, corpus=bow_corpus,
        dictionary=dictionary, coherence='u_mass')

text_to_file(str(coherence), os.path.join(output_dir, "evaluation.txt"))

console.success("Done!\n")
