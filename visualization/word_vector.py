import numpy as np

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


GRAPH_COLORS = [
    "#B6582F",
    "#2F8DB6",
    "#2FB685",
    "#B62F60",
    "#7241E4",
    "#B3E441",
    "#ABABAB",
    "#8D6E63",
    "#63828D",
    "#7EDE8E",
    "#DE7E7E",
]

GRAPH_MARKERS = [
    "o",
    "v",
    ">",
    "<",
    "^",
    "8",
    "s",
    "*",
    "d",
    "P",
    "X",
]


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

def plot_topic_words(
        topic_list, word2vec_model, save=False, 
        file_path=None, topic_ids=None):
    """Plots the topic words.

    Args:
        topic_list ([type]): [description]
        word2vec_model ([type]): [description]
        save (bool, optional): [description]. Defaults to False.
        file_path ([type], optional): [description]. Defaults to None.
        topic_ids ([type], optional): [description]. Defaults to None.
    """

    word_vectors_matrix = np.empty(
            (0, word2vec_model.vector_size), dtype=np.float64)
    word_list = []
    plot_configs = []
    topic_size = len(topic_list[0])
    tsne = TSNE(n_components=2, random_state=0)

    if topic_ids is None:
        topic_ids = list(range(len(topic_list)))

    for topic_id, topic in enumerate(topic_list):
        plot_configs.append({
            "c": GRAPH_COLORS[topic_id % len(GRAPH_COLORS)],
            "marker": GRAPH_MARKERS[topic_id % len(GRAPH_COLORS)],
        })

        word_vectors = np.empty((0, word2vec_model.vector_size), dtype=np.float64)
        for word in topic:
            word_list.append(word)

            word_vectors = np.append(
                    word_vectors, 
                    np.array([word2vec_model.get_vector(word)]),
                    axis=0)
        
        word_vectors_matrix = np.append(
                word_vectors_matrix,
                word_vectors, axis=0)

    plot_data = tsne.fit_transform(word_vectors_matrix)
    x_values = plot_data[:, 0]
    y_values = plot_data[:, 1]
    
    for topic_id in range(len(topic_list)):
        start = topic_id * topic_size
        end = start + topic_size

        plt.scatter(
                x_values[start:end], y_values[start:end], 
                label="Topic {}".format(topic_ids[topic_id]),
                **plot_configs[topic_id])

        for word, x, y in zip(
                word_list[start:end], x_values[start:end], 
                y_values[start:end]):
            plt.annotate(word, xy=(x, y))

    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.show()

    if save:
        plt.savefig(fname=file_path)