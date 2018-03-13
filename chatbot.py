import random

import models
import trainingdataset
from intents import Intents
from user_input import UserInput


def is_context_setter(intent):
    return 'context_set' in intent


def is_contextual(intent):
    return 'context_filter' in intent


def choose_response_for_intent(intent):
    return random.choice(intent['responses'])


class Chatbot:
    ERROR_THRESHOLD = 0.25
    # contains state for each user
    context = {}

    def __init__(self, file):
        self.intents = Intents(file)
        self.training_dataset = trainingdataset.TrainingDataset(self.intents)
        self.intents_dict = Intents().intents_dict
        self.model = models.TfModel(self.training_dataset.x_train, self.training_dataset.y_train)
        self.model.fit(self.training_dataset.x_train, self.training_dataset.y_train)

    def classify(self, sentence):
        inp = UserInput(sentence, lexicon=self.intents.lexicon)
        predictions = self.model.model.predict(inp.bag.reshape(1, len(self.intents.lexicon)))[0]
        # filter out predictions below a threshold
        predictions = [[i, p] for i, p in enumerate(predictions) if p > self.ERROR_THRESHOLD]
        prediction = max(predictions)
        # return tuple of prediction and probability
        return self.intents.classes[prediction[0]], prediction[1]

    def respond(self, sentence, userID=None):
        results = self.classify(sentence)
        pred_class = results[0]
        pred_intent = self.intents_dict[pred_class]
        self.set_context_for_further_conversation(pred_intent, userID)
        answer = self.apply_context(pred_intent, userID)
        return answer

    def set_context_for_further_conversation(self, intent, userID):
        if is_context_setter(intent):
            self.context[userID] = intent['context_set']

    def apply_context(self, intent, userID):
        if not is_contextual(intent):
            return choose_response_for_intent(intent)
        elif self.context_already_set(userID) and intent['context_filter'] == self.context[userID]:
            return choose_response_for_intent(intent)

    def context_already_set(self, userID):
        return userID in self.context
