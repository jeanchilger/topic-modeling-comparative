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

    word_vectors_matrix = []
    word_list = []
    # plot_configs = []
    plot_data = []
    tsne = TSNE(n_components=2, random_state=0)

    if topic_ids is None:
        topic_ids = list(range(len(topic_list)))

    for topic_id, topic in enumerate(topic_list):

        idx = topic_id % len(GRAPH_COLORS)
        plot_configs = {
            "c": GRAPH_COLORS[idx],
            "marker": GRAPH_MARKERS[idx],
        }

        word_vectors = np.empty((0, word2vec_model.vector_size), dtype=np.float64)
        for word in topic:
            word_list.append(word)

            word_vectors = np.append(
                    word_vectors, 
                    np.array([word2vec_model.get_vector(word)]),
                    axis=0)

        # print(word_vectors)
        # print()

        word_vectors_matrix.append(np.copy(word_vectors))

        plot_data.append(tsne.fit_transform(word_vectors_matrix[topic_id]))

        x_values = plot_data[topic_id][:, 0]
        y_values = plot_data[topic_id][:, 1]

        print(x_values)
        # print(x_values)
        print("--"*30)

        plt.scatter(
                x_values, y_values, 
                label="Topic {}".format(topic_ids[topic_id]),
                **plot_configs)

        for word, x, y in zip(topic, x_values, y_values):
            plt.annotate(word, xy=(x, y))

    print("*"*90)
    print(*word_vectors_matrix)


    plt.legend()
    plt.show()

    if save:
        plt.savefig(fname=file_path)