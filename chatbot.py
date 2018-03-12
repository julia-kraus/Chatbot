import random

import intents
import models
import trainingdataset
import user_input


class Chatbot:
    ERROR_THRESHOLD = 0.25

    def __init__(self, intentsfile):
        self.intents = intents.Intents(intentsfile)
        self.training_dataset = trainingdataset.TrainingDataset(self.intents)
        self.model = models.TfModel(self.training_dataset.x_train, self.training_dataset.y_train)
        self.model.fit(self.training_dataset.x_train, self.training_dataset.y_train)

    def classify(self, sentence):
        # convert sentence into userInput
        inp = user_input.UserInput(sentence, lexicon=self.intents.lexicon)
        results = self.model.model.predict(inp.bag.reshape(1, 40))[0]
        # filter out predictions below a threshold
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.intents.classes[r[0]], r[1]))
        # return tuple of intent and probability
        print(return_list)
        return return_list

    def response(self, sentence):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        for r in results:
            return print(random.choice(self.intents.responses(r)))
