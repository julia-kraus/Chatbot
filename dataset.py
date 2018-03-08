import pickle
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
        self.intents = intents
        self.x_train, self.y_train = self.get_training_data_from_intents(self.intents)

    def get_training_data_from_intents(self):
        features = self.get_features(self.intents)
        labels = self.get_labels(self.intents)
        return self.shuffle(features, labels)

    def get_features(self):
        features = []
        for doc in self.intents.documents:
            bag = build_bag_of_words(doc[0], self.intents.lexicon)
            features.append(bag)
        return np.array(features)

    def get_labels(self):
        """one hot encoding of the intent's class to a vector"""
        labels = []
        for doc in self.intents.documents:
            label = np.zeros(len(self.intents.classes))
            # for words that match in lexicon and document: set labels index to 1, else zero
            label[self.intents.classes.index(doc[1])] = 1
            labels.append(label)
        return np.array(labels)

    @staticmethod
    def shuffle(features, labels):
        training_data = list(zip(features, labels))
        random.shuffle(training_data)
        x_train, train_y = zip(*training_data)
        x_train = np.array(x_train)
        train_y = np.array(train_y)
        return x_train, train_y

    def save_data(self, intents):
        pickle.dump(
            {'lexicon': intents.lexicon, 'classes': intents.classes, 'x_train': self.x_train, 'y_train': self.y_train},
            open("training_data", "wb"))

    def load_data(self):
        data = pickle.load(open("training_data", "rb"))
