import numpy as np
from nltk.stem.lancaster import LancasterStemmer


def stem(words):
    """try to stem the words in user_intents.py"""
    stemmer = LancasterStemmer()
    ignore_words = ['?']
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    return words


def build_bag_of_words(words, lexicon):
    """input: lexicon and list of words. Return: bag of words"""
    bag = []
    words = stem(words)
    for w in lexicon:
        bag.append(1) if w in words else bag.append(0)
    return bag


class Dataset:
    # Class for creation of training- and test datasets
    # better: don't mingle dataset class
    x_train = []
    y_train = []

    def __init__(self, intents):
        self.x_train = self.get_features(intents)
        # self.y_train = self.get_labels

    @staticmethod
    def get_features(intents):
        features = []
        for doc in intents.documents:
            bag = build_bag_of_words(doc[0], intents.lexicon)
            features.append(bag)
        features = np.array(features)
        return features

    def get_labels(self, intents):
        # stimmt noch nicht ganz
        labels = []
        for doc in intents.documents:
            label = [intents.classes.index(doc[1])] == 1
            labels.append(label)
        return labels

    #
    # def create_training_data(self):
    #
    #     training_features = self.create_features
    #     training_labels = self.get_labels()
    #
    #     training_data = list(zip(training_features, training_labels))
    #
    #     random.shuffle(training)
    #
    #     return zip(*training_data)


def get_words_from_document(doc):
    words = doc[0]
    return words
