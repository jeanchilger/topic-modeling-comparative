from preprocessing.preprocess import Preprocessor
from utils.file_helpers import matrix_to_csv

file_names = [
        "only_phraser.csv",
        "merge_noun_chunks.csv",
        "merge_entities.csv",
        "both_merges.csv"
]

preprocessors = [
        Preprocessor(
                "data/news.csv",
                columns=["title", "description", "text"],
        ),

        Preprocessor(
                "data/news.csv",
                columns=["title", "description", "text"],
                merge_noun_chunks=True,
        ),

        Preprocessor(
                "data/news.csv",
                columns=["title", "description", "text"],
                merge_entities=True,
        ),

        Preprocessor(
                "data/news.csv",
                columns=["title", "description", "text"],
                merge_noun_chunks=True,
                merge_entities=True,
        )
]

for preprocessor, file_name in zip(preprocessors, file_names):
        print("writting {}...".format(file_name))
        matrix_to_csv(preprocessor.corpus, file_name)
        print("written successfully.")
        print()

