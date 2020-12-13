import sys
sys.path.insert(0, sys.path[0] + "/..")

from model.nmf import NmfModel
from preprocessing.preprocess import Preprocessor
from utils.file_helpers import csv_tokens_to_bow

from utils.visualization import vis_topic_distribution

data_files = [
    "only_phraser_3_gram_nohtml",
    "only_phraser_4_gram_nohtml",
    # "only_phraser_filtered_nohtml",
    # "both_merges_filtered_nohtml",
]

parameters = {
    "kappa": [0.4, 0.9],
    "minimum_probability": [0.1, 0.5],
}

# processor = Preprocessor("data/news.csv",
#         columns=["title", "description", "text"],
#         merge_entities=True)

# bow_corpus = processor.bag_of_words
# dictionary = processor.dictionary

corpora = []

for data_file in data_files:
    dictionary, bow_corpus = csv_tokens_to_bow(
            "data/preprocessed/" + data_file + ".csv")

    dictionary.filter_extremes(no_above=0.6)

    corpora.append((dictionary, bow_corpus, data_file))

for kappa in parameters["kappa"]:
    for mp in parameters["minimum_probability"]:
        # for normalize in parameters["normalize"]:
        for corpus in corpora:
            dictionary, bow_corpus, data_file = corpus

            model_name = "kappa={}, " + \
                    "minimum_probability={}, " + \
                    "data_file={}"
                    # "normalize={}"

            model_name = model_name.format(kappa, mp, data_file)

            print(model_name)
            model = NmfModel(
                    bow_corpus, num_topics=25,
                    kappa=kappa, minimum_probability=mp,
                    dictionary=dictionary)

            model.save(model_name)
            print("Done!\n")
