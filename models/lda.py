from gensim.models.ldamodel import LdaModel as Lda

RANDOM_STATE = 1

class LdaModel:

    def __init__(
            self, corpus=None, num_topics=10, dictionary=None,
            chunksize=1000, passes=200, alpha="symmmetric",
            eta=None, decay=0.5, iterations=50,
            model_name=None):

        if model_name is None:
            self._corpus = corpus
            self._num_topics = num_topics
            self._dictionary = dictionary

            self._model = Lda(
                    corpus, num_topics=num_topics, 
                    id2word=dictionary, chunksize=chunksize,
                    passes=passes, alpha=alpha,
                    eta=eta, decay=decay, iterations=iterations,
                    random_state=RANDOM_STATE)
                
        else:
            self._model = Lda.load(model_name)
            
            self._dictionary = self.model.id2word
            self._corpus = corpus
            self._num_topics = self.model.num_topics

    @property
    def model(self):
        return self._model
