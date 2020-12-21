import numpy as np
import pyLDAvis

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def vis_topic_distribution(model, corpus, dictionary):
    # corpus is bow_corpus
    
    topic_term_dists = model.get_topics()

    doc_topic_dists = [model.get_document_topics(document, normalize=True) 
            for document in corpus]

    doc_topic_dists = np.array(doc_topic_dists)

    print(doc_topic_dists[0])

    doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)[:, None]


    # docid = -1
    # for document in doc_topic_dists:
    #     docid += 1
    #     sum_all = 0
    #     for pair in document:
    #         sum_all += pair[1]

    #     if sum_all >= 1:
    #         print(sum_all)
    #         print(docid)

    doc_lengths = [len(doc) for doc in corpus]
    vocab = dictionary.token2id.keys()
    term_frequency = dictionary.cfs.values()

    data = {
        "topic_term_dists": topic_term_dists,
        "doc_topic_dists": doc_topic_dists,
        "doc_lengths": doc_lengths,
        "vocab": vocab,
        "term_frequency": term_frequency,
        "R": 10,
    }

    visualization_data = pyLDAvis.prepare(**data)
    pyLDAvis.show(visualization_data)


def plot_words(
        word_list, word2vec_model, save=False, 
        file_path=None, **scatter_kwargs):
    """Plots the given words using a scatter plot.

    Args:
        word_list (list of str): Words to be plotted.
        word2vec_model (Word2VecModel): Trainde word2vec model object.
    """

    word_vectors = np.empty((0, word2vec_model.vector_size), dtype=np.float64)

    for word in word_list:
        print(word)
        word_vectors = np.append(
                word_vectors, 
                np.array([word2vec_model.get_vector(word)]),
                axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    data = tsne.fit_transform(word_vectors)

    x_values = data[:, 0]
    y_values = data[:, 1]


    plt.scatter(x_values, y_values, **scatter_kwargs)

    for word, x, y in zip(word_list, x_values, y_values):
        plt.annotate(word, xy=(x, y))

    plt.show()

    if save:
        plt.savefig(fname=file_path)

    exit()