from gensim.models.nmf import Nmf

RANDOM_STATE=1

class NmfModel:

    def __init__(
            self, corpus=None, num_topics=10, dictionary=None, 
                chunksize=1000, passes=200, kappa=1.0,
                minimum_probability=0.01, normalize=False,
                model_name=None):

        if model_name is None:
            self._corpus = corpus
            self._num_topics = num_topics
            self._dictionary = dictionary

            self._model = Nmf(
                    corpus, num_topics=num_topics, 
                    id2word=dictionary, chunksize=chunksize,
                    passes=passes, kappa=kappa,
                    minimum_probability=minimum_probability,
                    normalize=normalize,
                    random_state=RANDOM_STATE)
                
        else:
            self._model = Nmf.load(model_name)

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

    def save(self, model_name):
        """
        Saves current model in trained_model folder.

        Args:
            model_name (string): name of model to be saved.
        """

        self.model.save(model_name)

    