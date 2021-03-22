import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import nltk
import pymorphy2
import re
import math
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
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
    data = pd.read_csv("processed_data.csv", encoding = 'ISO-8859-1')
    data = data[['v1', 'v2']]
    data['v1'] = (data['v1'] == 1).astype(int)
    r_texts = data['v2'].values
    r_labels = data['v1'].values
    return [r_texts, r_labels]


def splitter(s_text):
    return re.split(r'[\s,.?!]', s_text)


def create_vocabulary(texts):
    k = 1000
    all_tokens = []
    for v_text in texts:
        tokens = splitter(v_text)
        all_tokens = [*all_tokens, *list(filter(lambda t: t != '', tokens))]

    common = Counter(all_tokens).most_common(k)
    return list(map(lambda i: i[0], common))


def init_dictionary(vocab):
    vocab_dict = {}
    for token in vocab:
        vocab_dict[token] = 0

    return vocab_dict


def text_to_bow(text, vocab):
    text_vocab = init_dictionary(vocab)
    text_tokens = splitter(text)
    text_tokens = list(filter(lambda t: t != '', text_tokens))
    for token in text_tokens:
        if token in text_vocab:
            text_vocab[token] += 1

    return np.array(list(text_vocab.values()), 'float32')


class BinaryNaiveBayes:
    delta = 1.0  # add this to all word counts to smoothe probabilities

    def fit(self, X, y, vocab_length):
        """
        Fit a NaiveBayes classifier for two classes
        :param X: [batch_size, vocab_size] of bag-of-words features
        :param y: [batch_size] of binary targets {0, 1}
        """
        # 1 - spam
        # first, compute marginal probabilities of every class, p(y=k) for k = 0,1
        y_negative = list(filter(lambda label: label == 0, y))
        y_positive = list(filter(lambda label: label == 1, y))
        y_neg_prob = len(y_negative) / float(len(y))
        y_pos_prob = len(y_positive) / float(len(y))
        self.p_y = np.array([y_neg_prob, y_pos_prob])
        print(self.p_y)

        # count occurences of each word in texts with label 1 and label 0 separately
        word_counts_positive = 0
        word_counts_negative = 0
        for features, label in zip(X, y):
            if label == 1:
                word_counts_positive += features
            else:
                word_counts_negative += features
        # ^-- both must be vectors of shape [vocab_size].

        # finally, lets use those counts to estimate p(x | y = k) for k = 0, 1

        self.p_x_given_positive = (self.delta + word_counts_positive) / (np.sum(word_counts_positive) + vocab_length)
        self.p_x_given_negative = (self.delta + word_counts_negative) / np.sum(word_counts_negative + vocab_length)
        # both must be of shape [vocab_size]; and don't forget to add self.delta!

        return self

    def predict_scores(self, X):
        """
        :param X: [batch_size, vocab_size] of bag-of-words features
        :returns: a matrix of scores [batch_size, k] of scores for k-th class
        """
        # compute scores for positive and negative classes separately.
        # these scores should be proportional to log-probabilities of the respective target {0, 1}
        # note: if you apply logarithm to p_x_given_*, the total log-probability can be written
        # as a dot-product with X
        score_negative = []
        score_positive = []
        for x in X:
            score_negative.append(math.log(self.p_y[0]) + np.sum(np.log(self.p_x_given_negative) * x))
            score_positive.append(math.log(self.p_y[1]) + np.sum(np.log(self.p_x_given_positive) * x))


        # you can compute total p(x | y=k) with a dot product
        return np.stack([score_negative, score_positive], axis=-1)

    def predict(self, X):
        return self.predict_scores(X).argmax(axis=-1)


def check_naive_bayes(model, X_train_bow, X_test_bow, y_train, y_test):
    for name, X, y in [
        ('train', X_train_bow, y_train),
        ('test ', X_test_bow, y_test)
    ]:
        proba = model.predict_scores(X)[:, 1] - model.predict_scores(X)[:, 0]
        auc = roc_auc_score(y, proba)
        plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

    plt.plot([0, 1], [0, 1], '--', color='black',)
    plt.legend(fontsize='large')
    plt.grid()

    test_accuracy = np.mean(naive_model.predict(X_test_bow) == y_test)
    print(f"Model accuracy: {test_accuracy:.3f}")
    # assert test_accuracy > 0.75, "Accuracy too low. There's likely a mistake in the code."
    print("Well done!")


def check_logistic_regression(X_train, y_train, X_train_bow, X_test_bow, y_test):
    bow_model = LogisticRegression()
    bow_model.fit(X_train_bow, y_train)

    for name, X, y, model in [
        ('train', X_train_bow, y_train, bow_model),
        ('test ', X_test_bow, y_test, bow_model)
    ]:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

    plt.plot([0, 1], [0, 1], '--', color='black',)
    plt.legend(fontsize='large')
    plt.grid()

    test_accuracy = np.mean(bow_model.predict(X_test_bow) == y_test)
    print(f"Model accuracy: {test_accuracy:.3f}")
    # assert test_accuracy > 0.77, "Hint: tune the parameter C to improve performance"
    print("Well done!")


if __name__ == '__main__':
    [texts, labels] = read_data()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.4, random_state=42)
    bow_vocabulary = create_vocabulary(X_train)
    vocab_length = len(bow_vocabulary)
    to_bow = lambda x: text_to_bow(x, bow_vocabulary)
    X_train_bow = np.stack(list(map(to_bow, X_train)))
    X_test_bow = np.stack(list(map(to_bow, X_test)))

    k_max = len(set(' '.join(X_train).split()))
    print(len(X_train))

    naive_model = BinaryNaiveBayes().fit(X_train_bow, y_train, vocab_length)
    check_naive_bayes(naive_model, X_train_bow, X_test_bow, y_train, y_test)

    probability_ratio = naive_model.p_x_given_negative / naive_model.p_x_given_positive
    top_negative_words = sorted(zip(probability_ratio, bow_vocabulary))[-25:]

    for i, word in enumerate(top_negative_words):
        print(f"#{i}\t{word[1].rjust(10, ' ')}\t(ratio={probability_ratio[bow_vocabulary.index(word[1])]})")

    check_logistic_regression(X_train, y_train, X_train_bow, X_test_bow, y_test)




