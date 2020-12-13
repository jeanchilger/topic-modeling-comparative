import csv

from gensim.corpora import Dictionary

def csv_tokens_to_bow(input_file_path):
    """
    Reads a csv file and returns its BoW representation.
    Every column in the file must correspond to a token.

    Args:
        input_file_path (string): [description]
        skip_header (bool, optional): [description]. Defaults to False.

    Returns:
        Dictionary, bow: 
    """

    tokenized_corpus = []

    with open(input_file_path, "r") as src_csv:
        reader = csv.reader(src_csv)

        for row in reader:
            tokenized_corpus.append(row)

    dictionary = Dictionary(tokenized_corpus)
    dictionary.filter_extremes(no_above=0.6)
    bag_of_words = [dictionary.doc2bow(doc) for doc in tokenized_corpus]

    return dictionary, bag_of_words

def matrix_to_csv(matrix, output_file_path, header=None):
    """
    Writes the given matrix into an csv file.
    Each row in matrix corresponds to a row in the
    file. Optionally a list may be given as a header.

    Args:
        matrix (list of list): Input matrix to be written.
        output_file_path (str): Path to the destination file.
        header (list, optional): Header for the resulting csv. 
          Defaults to None.
    """

    with open(output_file_path, "w") as dest_csv:
        writer = csv.writer(dest_csv)

        if header:
            writer.writerow(header)

        for row in matrix:
            writer.writerow(row)

def matrix_to_txt(matrix, output_file_path, join_columns=True):
    """
    Writes the content of a matrix into a .txt file.

    Args:
        matrix (list of list): Input matrix to be written.
        output_file_path (str): Path to the destination file.
        join_columns (bool, optional): Whether or not join itens in rows.
          Defaults to True.
    """

    with open(output_file_path, "w") as dest_txt:
        for row in matrix:
            if join_columns:
                dest_txt.write(" ".join([str(item) for item in row]) + "\n")

            else:
                dest_txt.write(row + "\n")
