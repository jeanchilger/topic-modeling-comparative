from gensim.models.coherencemodel import CoherenceModel

def coherence_score(
        model=None, topics=None, texts=None, 
        corpus=None, dictionary=None, coherence='c_v'):
    """Wraps gensim.models.coherencemodel.CoherenceModel.

    Function to wrap gensim.models.coherencemodel.CoherenceModel
    coherence score generation. A model may be used if supported.

    Args:
        model (gensim BaseTopicModel, optional): Pre-trained topic model.
        topics (list of list of str, optional):  List of tokenized topics.
        texts (list of list of str, optional): Tokenized texts.
        corpus (iterable of list of (int, number), optional): 
            Corpus in BoW format.
        dictionary (gensim Dictionary, optional): Dictionary used to 
            generate corpus.
        coherence ({'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional): Coherence
            measure to be used.

    Returns:
        float: The coherence score.
    """

    coherence_model = None
    
    if model is None:
        coherence_model = CoherenceModel(
                topics=topics, corpus=corpus, 
                dictionary=dictionary, coherence=coherence)
    
    else:
        coherence_model = CoherenceModel(
                model=model, corpus=corpus, 
                coherence=coherence)

    return coherence_model.get_coherence()