import random

import models
import trainingdataset
from intents import Intents
from user_input import UserInput


class Chatbot:
    ERROR_THRESHOLD = 0.25
    # contains state for each user
    context = {}

    def __init__(self, file):
        self.ints = Intents(file)
        self.training_dataset = trainingdataset.TrainingDataset(self.ints)
        self.model = models.TfModel(self.training_dataset.x_train, self.training_dataset.y_train)
        self.model.fit(self.training_dataset.x_train, self.training_dataset.y_train)

    def classify(self, sentence):
        inp = UserInput(sentence, lexicon=self.ints.lexicon)
        results = self.model.model.predict(inp.bag.reshape(1, len(self.ints.lexicon)))[0]
        # filter out predictions below a threshold
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.ints.classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list

    def response(self, sentence, userID=None, show_details=True):
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        pred_classes = results[0]
        for c in pred_classes:
            answer = random.choice(self.ints.intents[c]["responses"])
            # set context for this intent if necessary
            if 'context_set' in self.ints.intents:
                if show_details:
                    print('context: ', self.ints.intents[c]['context_set'])
                self.context[userID] = self.ints.intents[c]['context_set']

            # # check if intent is contextual and apply to this user's conversation
            # if c not in self.intents.context_filter.keys() or (userID in self.context and ):
            #     if show_details: print('tag', )

            return answer
