import csv

import spacy
# spacy.cli.download("en_core_web_md")

import re

from gensim.corpora import Dictionary
from num2words import num2words


#################################################
# Insights
# 
# - remove HTML tags;
# - [X] convert numbers to text;
# - expand abbreviations;
# - named-entity recognition (NER) ?
# - coreference resolution
#################################################

class Preprocessor:
    """[summary]

    Description

    Attributes:
    """

    def __init__(
            self, file_path, columns,
            merge_noun_chunks=False, merge_entities=False,
            ngram=2):

        self._pipeline = spacy.load("en_core_web_md")
        self._ngram = ngram

        if merge_noun_chunks:
            noun_chunks_pipe = self._pipeline.create_pipe("merge_noun_chunks")
            self._pipeline.add_pipe(noun_chunks_pipe)

        if merge_entities:
            ents_pipe = self._pipeline.create_pipe("merge_entities")
            self._pipeline.add_pipe(ents_pipe)
            
        
        self._raw_text = self._get_csv_contents(file_path, columns)
        self._joined_text = self._concat_raw_text()
        self._corpus = self._tokenize()

        self._ngram_corpus = None
        self._bag_of_words = None
        self._bag_of_ngrams = None

    @property
    def bag_of_words(self):
        if self._bag_of_words is None:
            self._dictionary = Dictionary(self.corpus)
            self._bag_of_words = self._generate_bow()

        return self._bag_of_words

    @property
    def bag_of_ngrams(self):
        if self._bag_of_ngrams is None:
            self._ngram_dictionary = Dictionary(self.ngram_corpus)
            self._bag_of_ngrams = self._generate_bag_of_ngrams()

        return self._bag_of_ngrams

    @property
    def corpus(self):
        return self._corpus

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def joined_text(self):
        return self._joined_text

    @property
    def ngram_corpus(self):
        if self._ngram_corpus is None:
            self._ngram_corpus = self._generate_ngrams(n=self._ngram)

        return self._ngram_corpus

    @property
    def raw_text(self):
        return self._raw_text


    def _concat_raw_text(self):
        """
        Concatenates raw_text columns into a single text.
        The result will join all columns (given at init) 
        into a single string.

        Returns:
            list: list with the result text in every entry.
        """

        assert self.raw_text

        corpus = []
        for row in self.raw_text:
            corpus.append(" ".join(row))

        return corpus

    def _generate_ngrams(self, n=2):
        """[summary]

        Args:
            n (int, optional): Size of the n-gram model.
              Defaults to 2.

        Returns:
            list: The n-gram model.
        """

        ngram_corpus = []
        doc_idx = -1
        for document in self.corpus:
            ngram_corpus.append([])
            doc_idx += 1
            for i in range(len(document) - (n - 1)):
                ngram_corpus[doc_idx].append(" ".join(document[i:i + n]))

        return ngram_corpus

    def _generate_bow(self):
        """[summary]

        Returns:
            [type]: BOW representation of corpus.
        """

        # self._dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        word_bag = [self.dictionary.doc2bow(doc) for doc in self.corpus]

        return word_bag

    def _generate_bag_of_ngrams(self):
        """[summary]

        Returns:
            [type]: BOW representation of corpus.
        """

        # self._dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        ngram_bag = [self.dictionary.doc2bow(doc) for doc in self.ngram_corpus]

        return ngram_bag

    def _get_csv_contents(self, file_path, columns, max_docs=100):
        """
        Reads a csv file and its contents for desired columns.

        Args:
            file_path (string): path to csv file.
            columns (list): list with desired columns' names.

        Returns:
            list: returns a matrix, every row corresponding to
                  a row from the file.
        """

        n_docs = 0
        raw_text = []
        column_indexes = []
        with open(file_path, "r") as data_file:
            csv_reader = csv.reader(data_file)

            # sets up only desired columns
            header = list(next(csv_reader))
            
            for column in columns:
                column_indexes.append(header.index(column))

            for row in csv_reader:
                n_docs += 1
                raw_text.append([row[i] for i in column_indexes])

                if n_docs >= max_docs:
                    break

        return raw_text

    def _is_token_valid(self, token):
        """
        Returns whether or not a token is a valid one.

        Args:
            token (Token): a token from document.

        Returns:
            boolean: true if token is valid, false otherwise.
        """

        return not token.is_stop and not token.is_punct

    def _lemmatize(self, token):
        """
        Obtains the lemma for the given token, performing
        some adjustments, such as digit to word.

        Args:
            token (Token): Token to lemmatize.

        Returns:
            string: Lemma of the given token.
        """

        lemma = token.lemma_

        if token.is_digit:
            lemma = " ".join(re.split(", ", num2words(lemma)))

        return lemma


    def _tokenize(self):
        """
        Returns the tokens for the corpus.
        Tokens are already lemmatized, without stop words.

        Returns:
            list: tokens, every entry is a list of tokens.
        """

        tokens = []
        for text in self._joined_text:
            document = self._pipeline(text)
            
            tokens.append([
                self._lemmatize(token)
                for token in document 
                if self._is_token_valid(token)
            ])

        return tokens
