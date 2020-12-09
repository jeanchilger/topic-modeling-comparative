import csv
import spacy
# spacy.cli.download("en_core_web_md")
import re

from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser

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
        raw_text:

        joined_text:

        corpus:

        dictionary:

        bag_of_words:

        phrases:
    """

    def __init__(
            self, file_path, columns,
            merge_noun_chunks=False, merge_entities=False,
            ngram=2, keep_digits=False):

        self._pipeline = spacy.load("en_core_web_md")
        self._keep_digits = keep_digits

        if merge_noun_chunks:
            noun_chunks_pipe = self._pipeline.create_pipe("merge_noun_chunks")
            self._pipeline.add_pipe(noun_chunks_pipe)

        if merge_entities:
            ents_pipe = self._pipeline.create_pipe("merge_entities")
            self._pipeline.add_pipe(ents_pipe)

        self._use_phraser = not (merge_noun_chunks or merge_entities)
        
        self._raw_text = self._get_csv_contents(file_path, columns)
        self._joined_text = self._concat_raw_text()

        self._corpus = self._tokenize()
        if self._use_phraser:
            self._corpus = self._detect_phrases(min_count=2)

        self._bag_of_words = None

    @property
    def bag_of_words(self):
        if self._bag_of_words is None:
            self._dictionary = Dictionary(self.corpus)
            self._bag_of_words = self._generate_bow()

        return self._bag_of_words

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

    def _detect_phrases(self, min_count):
        """
        Performs the phrase (collocation) detection.
        Detected bigrams are joined.

        Args:
            min_count (int): [description]

        Returns:
            list: Tokens with joined detected phrases.
        """

        # connector_words
        phrases = Phrases(self.corpus, min_count=min_count)
        frozen_phrases = Phraser(phrases)

        return [frozen_phrases[token] for token in self.corpus]

    def _generate_bow(self):
        """
        Returns the BoW representation of inner corpus.
        The dictionary is obtained based on this corpus
        also.

        Returns:
            list: BoW representation of corpus.
        """

        bag_of_words = [self.dictionary.doc2bow(doc) for doc in self.corpus]

        return bag_of_words

    def _get_csv_contents(self, file_path, columns):
        """
        Reads the contents from a csv file.
        Only the given columns are read.

        Args:
            file_path (string): path to csv file.
            columns (list): list with desired columns' names.

        Returns:
            list: returns a matrix, every row corresponding to
              a row from the file.
        """

        # For tests
        n_docs = 20
        idx = 0

        raw_text = []
        column_indexes = []
        with open(file_path, "r") as data_file:
            csv_reader = csv.reader(data_file)

            # sets up only desired columns
            header = list(next(csv_reader))
            
            for column in columns:
                column_indexes.append(header.index(column))

            for row in csv_reader:
                # For tests
                idx += 1
                if idx >= n_docs:
                    break
                
                raw_text.append([row[i] for i in column_indexes])

        return raw_text

    def _is_token_valid(self, token):
        """
        Returns whether or not a token is a valid one.

        Args:
            token (Token): a token from document.

        Returns:
            boolean: true if token is valid, false otherwise.
        """

        return not (token.is_stop or token.is_punct or \
                token.is_space or (not self._keep_digits and token.is_digit))

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

        if self._keep_digits and token.is_digit:
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
