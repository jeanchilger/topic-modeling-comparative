import pyLDAvis

def vis_topic_distribution(model, corpus, dictionary):
    
    topic_term_dists = model.get_topics()
    doc_topic_dists = [model.get_document_topics(document) 
            for document in corpus]
    doc_lengths = [len(doc) for doc in corpus]
    vocab = dictionary.token2id.keys()
    term_frequency = cfs.values()

    data = {
        "topic_term_dists": topic_term_dists,
        "doc_topic_dists": doc_topic_dists,
        "doc_lengths": doc_lengths,
        "vocab": vocab,
        "term_frequency": term_frequency,
    }

    visualization_data = pyLDAvis.prepare(**data)
    pyLDAvis.show(visualization_data)