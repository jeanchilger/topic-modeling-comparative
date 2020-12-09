from gensim.models.nmf import Nmf

RANDOM_STATE=1

class NmfModel:

    def __init__(
            self, corpus, num_topics, dictionary=None, 
            chunksize=1000, passes=200, kappa=1.0):
        self._corpus = corpus
        self._num_topics = num_topics
        self._dictionary = dictionary

        self._model = Nmf(
                corpus, num_topics=num_topics, 
                id2word=dictionary, chunksize=chunksize,
                passes=passes, kappa=kappa,
                random_state=RANDOM_STATE)

    @property
    def model(self):
        return self._model

    def get_document_topics(self, document):
        """
        Wraps the get_document_topics from gensim's Nmf.
        Returns the topic distribution for a given
        document.

        Args:
            document (list of (int, int), bow): [description]

        Returns:
            [type]: [description]
        """
        return self.model.get_document_topics(document)