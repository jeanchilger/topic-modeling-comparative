import numpy as np
import pyLDAvis


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
