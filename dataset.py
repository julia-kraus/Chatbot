import random

import numpy as np


def build_bag_of_words(words, lexicon):
    """input: lexicon and list of words. Return: bag of words"""
    bag = []
    for w in lexicon:
        bag.append(1) if w in words else bag.append(0)
    return bag


class Dataset:
    # Class for creation of training- and test datasets from intents or user input
    # New class for test data, because it only needs get_features

    def __init__(self, intents):
        self.x_train, self.y_train = self.get_training_data_from_intents(intents)

    def get_training_data_from_intents(self, intents):
        features = self.get_features(intents)
        labels = self.get_labels(intents)
        return self.shuffle(features, labels)

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
            # for words that match in lexicon and document: set labels index to 1, else zero
            label[intents.classes.index(doc[1])] = 1
            labels.append(label)
        return np.array(labels)

    @staticmethod
    def shuffle(features, labels):
        training_data = list(zip(features, labels))
        random.shuffle(training_data)
        train_x, train_y = zip(*training_data)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return train_x, train_y
