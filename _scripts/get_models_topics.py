import sys
sys.path.insert(0, sys.path[0] + "/..")

import os

from gensim.models.nmf import Nmf
from utils.file_helpers import matrix_to_txt

for model_file_name in os.listdir("trained_models"):
    
    model = Nmf.load(os.path.join("trained_models", model_file_name))

    topics = model.show_topics(num_topics=25, formatted=True)

    matrix_to_txt(topics, "results/nmf/topics/" + model_file_name)