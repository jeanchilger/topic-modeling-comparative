import sys
sys.path.insert(0, sys.path[0] + "/..")

from preprocessing.preprocess import Preprocessor
from utils.file_helpers import matrix_to_csv
from utils.file_helpers import matrix_to_txt

file_names = [
    "only_phraser",
    "only_phraser_with_digits",
    "merge_noun_chunks",
    "merge_entities",
    "both_merges",
    "both_merges_with_digits"
]

preprocessors = [
    Preprocessor(
            "../data/news.csv",
            columns=["title", "description", "text"]),

    Preprocessor(
            "../data/news.csv",
            columns=["title", "description", "text"],
            keep_digits=True),

    Preprocessor(
            "../data/news.csv",
            columns=["title", "description", "text"],
            merge_noun_chunks=True),

    Preprocessor(
            "../data/news.csv",
            columns=["title", "description", "text"],
            merge_entities=True),

    Preprocessor(
            "../data/news.csv",
            columns=["title", "description", "text"],
            merge_noun_chunks=True,
            merge_entities=True),
    
    Preprocessor(
            "../data/news.csv",
            columns=["title", "description", "text"],
            merge_noun_chunks=True,
            merge_entities=True,
            keep_digits=True),
]

for preprocessor, file_name in zip(preprocessors, file_names):
    print("writting {}...".format(file_name))

    matrix_to_csv(
            preprocessor.corpus, 
            "../data/preprocessed/" + file_name + ".csv")
    
    matrix_to_csv(
            preprocessor.corpus, 
            "../data/preprocessed/" + file_name + ".txt")

    print("written successfully.")
    print()

