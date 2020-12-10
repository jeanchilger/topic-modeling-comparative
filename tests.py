from model.nmf import NmfModel
from preprocessing.preprocess import Preprocessor
from utils.file_helpers import csv_tokens_to_bow

# from utils.visualization import vis_topic_distribution

input_file_path = "data/preprocessed/only_phraser_with_digits.csv"

# parameters = {
#     "kappa": [0.4, 0.9, 1.5],
#     "minimum_probability": [0.005, 0.02, 0.1],
#     "normalize": [True, False],
# }

processor = Preprocessor("data/news.csv",
        columns=["title", "description", "text"],
        keep_digits=True,
        merge_entities=True)

corpus = processor.corpus
bow_corpus = processor.bag_of_words
dictionary = processor.dictionary

print(len(corpus[0]))
print(corpus[0])