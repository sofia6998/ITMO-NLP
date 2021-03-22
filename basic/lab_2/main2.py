import re
import nltk
from nltk.corpus import stopwords
import pymorphy2
from collections import Counter
from nltk.collocations import *
from functools import cmp_to_key
import math

_SMALL = 1e-20

# nltk.download('stopwords')


def get_lems(tokens_array):
    morph = pymorphy2.MorphAnalyzer(lang='ru')
    res = []
    for t in tokens_array:
        p = morph.parse(t)[0]
        res.append(p.normal_form)

    return res


def remove_stop_words(in_tokens):
    stop_words = set(stopwords.words('russian'))
    filtered_words = list(filter(lambda x: not x in stop_words, in_tokens))
    return filtered_words


def get_prepeared_tokens():
    f = open("kabluk.txt", "r")
    text = f.read()
    f.close()
    dirty_tokens = re.split(r'[^a-zA-Z0-9А-Яа-яёЁ]', text)
    filtered_tokens = list(filter(lambda x: len(x) > 0, dirty_tokens))
    lems = get_lems(filtered_tokens)
    lower_lems = list(map(lambda x: x.lower(), lems))
    return remove_stop_words(lower_lems)


def search_trigrams(in_tokens):
    trigrams_arr = []
    full_tokens = ['<start>', *in_tokens, '<end>']
    for x in range(0, len(full_tokens) - 2):
        trigrams_arr.append((full_tokens[x], full_tokens[x + 1], full_tokens[x + 2]))

    return Counter(trigrams_arr)


def calc_association_scores(in_trigrams):
    scores = []
    for tr in in_trigrams:
        scores.append((calc_association_score(tr, in_trigrams), tr))
    return scores


def calc_association_score(trigram, all_trigrams):
    o = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    e = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    n_ppp = 0
    n = [[0,0],[0,0], [0,0]]

    for t in all_trigrams:
        for x in range(0, 2):
            for y in range(0, 2):
                for z in range(0, 2):
                    is_first_mathch = (trigram[0] == t[0]) == (x == 0)
                    is_second_match = (trigram[1] == t[1]) == (y == 0)
                    is_third_match = (trigram[2] == t[2]) == (z == 0)
                    if is_first_mathch and is_second_match and is_third_match:
                        o[x][y][z] += all_trigrams[t]

        n_ppp += all_trigrams[t]

        n = [
            [
                o[0][0][0] + o[0][0][1] + o[0][1][0] + o[0][1][1],
                o[1][0][0] + o[1][0][1] + o[1][1][0] + o[1][1][1],
                ],
            [
                o[0][0][0] + o[0][0][1] + o[1][0][0] + o[1][0][1],
                o[0][1][0] + o[0][1][1] + o[1][1][0] + o[1][1][1],
                ],
            [
                o[0][0][0] + o[0][1][0] + o[1][0][0] + o[1][1][0],
                o[0][0][1] + o[0][1][1] + o[1][0][1] + o[1][1][1],
                ]
        ]

    n_ooo = o[1][1][1]
    suma = 0
    for x in range(0, 2):
        for y in range(0, 2):
            for z in range(0, 2):
                e[x][y][z] = (n[0][x] * n[1][y] * n[2][z]) / (n_ppp * n_ppp)
                suma += o[x][y][z] * math.log(o[x][y][z] / (e[x][y][z] + _SMALL) + _SMALL)

    score = 3 * suma

    # print(trigram, o, e, n, n_ppp)
    return score

if __name__ == '__main__':
    tokens = get_prepeared_tokens()
    trigrams = search_trigrams(tokens)
    calc_scores = calc_association_scores(trigrams)
    f = open("my.txt", "a")

    def compare(x, y):
        if x[0] == y[0]:
            return x[1][0] < y[1][0]
        return y[0] - x[0]

    sorted_scores = sorted(calc_scores, key=lambda tup: (-tup[0], tup[1][0]))
    for res in sorted_scores:
        f.write(str(res[0]) + ":" + res[1][0] + " " + res[1][1] + " " + res[1][2] + "\n")
    f.close()

    # check
    text = nltk.Text(tokens)
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    #http://www.nltk.org/_modules/nltk/collocations.html
    finder_bi = BigramCollocationFinder.from_words(text)
    finder_thr = TrigramCollocationFinder.from_words(text)
    #
    # print(finder_thr.nbest(trigram_measures.likelihood_ratio, 10))
    likelihood = finder_thr.nbest(trigram_measures.likelihood_ratio, 100)
    fl = open("likelihood.txt", "a")
    for res in likelihood:
        fl.write(str(res[1]) + ": " + res[0][0] + " " + res[0][1] + " " + res[0][2] + " " + "\n")
    fl.close()

    mi = finder_thr.nbest(trigram_measures.mi_like, 100)
    fm = open("mi.txt", "a")
    for res in mi:
        fm.write(res[0] + " " + res[1] + " " + res[2] + " " + "\n")
    fm.close()

    pmi = finder_thr.nbest(trigram_measures.pmi, 100)
    fp = open("pmi.txt", "a")
    for res in pmi:
        fp.write(res[0] + " " + res[1] + " " + res[2] + " " + "\n")
    fp.close()

