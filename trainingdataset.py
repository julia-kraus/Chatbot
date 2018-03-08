import pickle
import random

import numpy as np

import word_utils


class TrainingDataset:
    # Class for creation of training- and test datasets from intents or user input
    # New class for test data, because it only needs get_features

    def __init__(self, intents):
        self.intents = intents
        self.x_train, self.y_train = self.get_training_data_from_intents()

    def get_training_data_from_intents(self):
        features = self.get_features()
        labels = self.get_labels()
        return self.shuffle(features, labels)

    def get_features(self):
        features = []
        for doc in self.intents.documents:
            bag = word_utils.build_bag_of_words(doc[0], self.intents.lexicon)
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
        x_train, y_train = zip(*training_data)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

    def save_data(self):
        with open("training_data", "wb") as file:
            pickle.dump(
                {'lexicon': self.intents.lexicon, 'classes': self.intents.classes, 'x_train': self.x_train,
                 'y_train': self.y_train},
                file)

    @staticmethod
    def load_data():
        with open("training_data", "rb") as file:
            data = pickle.load(file)
        lexicon = data['lexicon']
        classes = data['classes']
        x_train = data['x_train']
        y_train = data['y_train']
        return lexicon, classes, x_train, y_train
