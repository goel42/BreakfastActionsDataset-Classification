import numpy as np


def load_word2vec():
    # TODO: path and dimensions could be params or passed as arguments for flexibility
    embeddings_dict = {}
    path = r"C:\Users\dcsang\PycharmProjects\embedding\Mixture-of-Embedding-Experts\word2vec\vectors_200d.txt"
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict
