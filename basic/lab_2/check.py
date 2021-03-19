import nltk
import re
from nltk.collocations import *
from nltk.corpus import PlaintextCorpusReader


if __name__ == '__main__':
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    f = open('kabluk.txt')
    raw = f.read()

    dirty_tokens = nltk.word_tokenize(raw, 'russian', True)
    tokens = list(filter(lambda x: not re.match(r'[^a-zA-Z0-9А-Яа-яёЁ]', x), dirty_tokens))
    print(tokens[:100])

    text = nltk.Text(tokens)

    #http://www.nltk.org/_modules/nltk/collocations.html
    finder_bi = BigramCollocationFinder.from_words(text)
    finder_thr = TrigramCollocationFinder.from_words(text)

    # f = open("errors.txt", "a")
    print(finder_bi.nbest(bigram_measures.pmi, 10))
    print(finder_thr.nbest(trigram_measures.likelihood_ratio, 10))
    # f.write(finder_thr.nbest(trigram_measures.pmi, 100))
    # f.close()

