import sys
sys.path.insert(0, sys.path[0] + "/..")
import utils.console as console

from models.nmf import NmfModel
from preprocessing.preprocess import Preprocessor
from utils.file_helpers import csv_tokens_to_bow
from utils.visualization import vis_topic_distribution


data_files = [
    "only_phraser",
    "only_phraser_with_digits",
    "both_merges",
    "both_merges_with_digits",
]

parameters = {
    "kappa": [0.09, 0.4, 0.9],
    "minimum_probability": [0.005, 0.5],
}

corpora = []

for data_file in data_files:
    dictionary, bow_corpus = csv_tokens_to_bow(
            "data/preprocessed/" + data_file + ".csv")

    dictionary.filter_extremes(no_above=0.6)

    corpora.append((dictionary, bow_corpus, data_file))

for kappa in parameters["kappa"]:
    for mp in parameters["minimum_probability"]:
        for corpus in corpora:
            dictionary, bow_corpus, data_file = corpus

            model_name = "kappa={}, " + \
                    "minimum_probability={}, " + \
                    "data_file={}"

            model_name = model_name.format(kappa, mp, data_file)

            console.info(model_name)
            model = NmfModel(
                    bow_corpus, num_topics=25,
                    kappa=kappa, minimum_probability=mp,
                    dictionary=dictionary)

            model.save(model_name)
            console.success("Done!\n")
