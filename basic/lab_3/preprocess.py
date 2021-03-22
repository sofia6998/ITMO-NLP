import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk
import pymorphy2
import re
import math
import csv
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve
nltk.download('punkt')
# %matplotlib inline

import pandas as pd

NUMBER_REPLACES = '<NUMBER>'


# another dataset for sentiment analysis of movie reviews

# train_data = pd.read_csv("imdb_reviews/train.csv")
# test_data = pd.read_csv("imdb_reviews/test.csv")
# texts_train = train_data['reviews'].values
# y_train = train_data['sentiment'].values
# texts_test = test_data['reviews'].values
# y_test = test_data['sentiment'].values

def read_data():
    data = pd.read_csv("spam.csv", encoding = 'ISO-8859-1')
    data = data[['v1', 'v2']]
    data['v1'] = (data['v1'] == 'spam').astype(int)
    r_texts = data['v2'].values
    r_labels = data['v1'].values
    return [r_texts, r_labels]


def get_lems(token):
    morph = pymorphy2.MorphAnalyzer(lang='ru')
    res = []
    p = morph.parse(token)[0]
    return NUMBER_REPLACES if is_number(p.normal_form) else p.normal_form


def preprocess(text):
    tokens = filter(lambda x: not re.fullmatch(r'[,.?!]+', x), word_tokenize(text.lower()))
    lems = list(map(get_lems, tokens))
    return ' '.join(lems)


def splitter(s_text):
    return re.split(r'[\s,.?!]', s_text)


def is_number(n_token):
    return bool(re.match(r'\d+', n_token))


if __name__ == '__main__':
    [texts, labels] = read_data()
    with open('full_processed_data.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        i = 0
        for text, label in zip(texts, labels):
            spamwriter.writerow(["\"" + preprocess(text) + "\"", label])
            print(i)
            i+=1
