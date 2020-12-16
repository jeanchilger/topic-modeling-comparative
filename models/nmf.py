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
            
            self._dictionary = self.model.id2word
            self._corpus = corpus
            self._num_topics = self.model.num_topics

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

    def get_common_topics(self, topn=10, thresh=3):
        """Returns the topics with most documents related to.

        Gets a list of topn topics ordered by the frequency of 
        documents related to them. For every document, the top thresh
        topics are used to calculate the frequency.

        Args:
            topn (int, optional): [description]. Defaults to 10.
            thresh (int, optional): [description]. Defaults to 3.

        Returns:
            [type]: [description]
        """

        topic_counts = {}

        for i in range(self._num_topics):
            topic_counts[i] = 0

        for doc_bow in self._corpus:
            _thresh = 0
            for topic_id, score in sorted(
                    self.get_document_topics(doc_bow), 
                    key=lambda tup: -1*tup[1]):

                _thresh += 1

                if _thresh > thresh:
                    break

                topic_counts[topic_id] += 1

        topic_counts = dict(sorted(
                topic_counts.items(), 
                key=lambda item: item[1]))

        _topn = 0
        top_topics = []
        for topic_id in topic_counts.keys():
            _topn += 1
            if _topn > topn:
                break
            top_topics.append(self.model.show_topic(topic_id))

        return top_topics


    def save(self, model_name):
        """
        Saves current model in trained_model folder.

        Args:
            model_name (string): name of model to be saved.
        """

        self.model.save(model_name)

    