from gensim.models import KeyedVectors
from gensim.test.utils import datapath


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


if __name__ == '__main__':
    wv_from_bin = load_word2vec()
    vocab = list(wv_from_bin.vocab.keys())
    for v in vocab:
        guesses = wv_from_bin.most_similar(v)
        print(".")
        for g in guesses:
            for gg in guesses:
                if g[1] > 0.5 and gg[1] > 0.5 and g[0] != gg[0]:
                    sim = wv_from_bin.similarity(g[0], gg[0])
                    if sim < 0.1:
                        print(v, ":", g, gg,  "____", sim)