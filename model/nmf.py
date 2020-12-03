from gensim.models.nmf import Nmf

class NmfModel:

    def __init__(self, corpus, n_topics, id2word=None):
        self.corpus = corpus