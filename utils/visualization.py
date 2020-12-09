import pyLDAvis.gensim

def vis_topic_distribution(model, corpus, dictionary):
    visualization_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)

    pyLDAvis.show(visualization_data)