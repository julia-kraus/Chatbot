import random

import numpy as np


def build_bag_of_words(words, lexicon):
    """input: lexicon and list of words. Return: bag of words"""
    bag = []
    for w in lexicon:
        bag.append(1) if w in words else bag.append(0)
    return bag


class Dataset:
    # Class for creation of training- and test datasets
    # creates training or test data from intents
    x_train = []
    y_train = []

    def __init__(self, intents):
        self.x_train = self.get_features(intents)
        self.y_train = self.get_labels(intents)

    @staticmethod
    def get_features(intents):
        features = []
        for doc in intents.documents:
            bag = build_bag_of_words(doc[0], intents.lexicon)
            features.append(bag)
        return np.array(features)

    @staticmethod
    def get_labels(intents):
        """one hot encoding of the intent's class to a vector"""
        labels = []
        for doc in intents.documents:
            label = np.zeros(len(intents.classes))
            label[intents.classes.index(doc[1])] = 1
            labels.append(label)
        return np.array(labels)

    @staticmethod
    def get_training_data(features, labels):
        training_data = list(zip(features, labels))
        random.shuffle(training_data)
        return zip(*training_data)
