from model.nmf import NmfModel
from preprocessing.preprocess import Preprocessor
from sklearn.model_selection import GridSearchCV
from utils.file_helpers import csv_tokens_to_bow

# from utils.visualization import vis_topic_distribution

# input_file_path = "data/preprocessed/both_merges.csv"

processor = Preprocessor("data/news.csv",
        columns=["title", "description", "text"])

# dictionary, bow_corpus = csv_tokens_to_bow(input_file_path)

bow_corpus = processor.bag_of_words
dictionary = processor.dictionary

# model = NmfModel(bow_corpus, num_topics=25, dictionary=dictionary)



print(dictionary.token2id["say"])
# print(dictionary["say"])
print(dictionary.doc2bow(["say"]))

# print(processor.corpus[0])
# print(processor.corpus[1])
# for topic in model.model.show_topics():
#     print(topic)
# vis_topic_distribution(model.model, bow_corpus, dictionary)