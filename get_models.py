from model.nmf import NmfModel
from preprocessing.preprocess import Preprocessor
from utils.file_helpers import csv_tokens_to_bow

from utils.visualization import vis_topic_distribution

input_file_path = "data/preprocessed/only_phraser.csv"

parameters = {
    "kappa": [0.4, 0.9, 1.5],
    "minimum_probability": [0.005, 0.02, 0.1],
    "normalize": [True, False],
}

# processor = Preprocessor("data/news.csv",
#         columns=["title", "description", "text"],
#         merge_entities=True)

# bow_corpus = processor.bag_of_words
# dictionary = processor.dictionary

dictionary, bow_corpus = csv_tokens_to_bow(input_file_path)

for kappa in parameters["kappa"]:
    for mp in parameters["minimum_probability"]:
        for normalize in parameters["normalize"]:
            model_name = "kappa={}, " + \
                    "minimum_probability={}, " + \
                    "normalize={}"

            model_name = model_name.format(kappa, mp, normalize)

            print(model_name)
            model = NmfModel(
                    bow_corpus, num_topics=25,
                    kappa=kappa, minimum_probability=mp,
                    normalize=normalize,
                    dictionary=dictionary)

            model.save(model_name)
            print("Done!\n")
