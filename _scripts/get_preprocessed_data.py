import sys
sys.path.insert(0, sys.path[0] + "/..")
import utils.console as console

from preprocessing.preprocess import Preprocessor
from utils.file_helpers import matrix_to_csv
from utils.file_helpers import matrix_to_txt


file_names = [
    "only_phraser",
    "only_phraser_3_gram",
    "only_phraser_4_gram",
    "only_phraser_with_digits",
    "merge_noun_chunks",
    "merge_entities",
    "both_merges",
    "both_merges_with_digits"
]

preprocessor_configs = [
    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "ngram": 3,
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "ngram": 4,
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "keep_digits": True,
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "merge_noun_chunks": True,
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "merge_entities": True,
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "merge_noun_chunks": True,
        "merge_entities": True,
    },

    {
        "file_path": "data/news.csv",
        "columns": ["title", "description", "text"],
        "keep_digits": True,
        "merge_noun_chunks": True,
        "merge_entities": True,
    },
]


for config, file_name in zip(preprocessor_configs, file_names):
    console.info("writting {}...".format(file_name))

    preprocessor = Preprocessor(**config)

    matrix_to_csv(
            preprocessor.corpus,
            "data/preprocessed/" + file_name + ".csv")

    matrix_to_csv(
            preprocessor.corpus,
            "data/preprocessed/" + file_name + ".txt")

    console.success("written successfully.")
    print()

