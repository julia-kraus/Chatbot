import intents
import models
import trainingdataset
import user_input


class Chatbot:
    ERROR_THRESHOLD = 0.25
    # contains state for each user
    context = {}

    def __init__(self, intentsfile):
        self.intents = intents.Intents(intentsfile)
        self.training_dataset = trainingdataset.TrainingDataset(self.intents)
        self.model = models.TfModel(self.training_dataset.x_train, self.training_dataset.y_train)
        self.model.fit(self.training_dataset.x_train, self.training_dataset.y_train)

    def classify(self, sentence):
        # convert sentence into userInput
        inp = user_input.UserInput(sentence, lexicon=self.intents.lexicon)
        results = self.model.model.predict(inp.bag.reshape(1, len(self.intents.lexicon)))[0]
        # filter out predictions below a threshold
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.intents.classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list

    # def response(self, sentence, userID, show_details):
    #     results = self.classify(sentence)
    #     # if we have a classification then find the matching intent tag
    #     pred_classes = results[0]
    #     for c in pred_classes:
    #         answer = random.choice(self.intents.responses[c])
    #         # set context for this intent if necessary
    #         if c in self.intents.context_set.keys():
    #             if show_details:
    #                 print('context: ', self.intents.context_set[c])
    #                 self.context[userID] = self.intents.context_set[c]
    #
    #         # check if intent is contextual and apply to this user's conversation
    #         if c not in self.intents.context_filter.keys() or (userID in self.context and ):
    #             if show_details: print('tag', )
    #
    #
    #         return answer
