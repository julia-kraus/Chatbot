import unittest

import models
import trainingdataset
import user_input
import user_intents


class UserInputTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def test_one_sentence(self):
        inp = user_input.UserInput("is there a cheap medication?", lexicon)
        self.assertEqual(inp.sentence, "is there a cheap medication?")
        self.assertEqual(inp.words, ["is", "ther", "a", "cheap", "med"])
        self.assertEqual(inp.bag, [0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 0, 1, 0, 0, 0, 0, 0])


lexicon = ["'m", "'s", 'a', 'anyon', 'ar', 'buy', 'bye', 'can', 'cheap', 'cheapest',
           'coupon', 'day', 'deal', 'find', 'for', 'good', 'goodby', 'hello', 'help', 'hi',
           'how', 'i', 'is', 'lat', 'less', 'look', 'me', 'med', 'money', 'see',
           'send', 'thank', 'that', 'the', 'ther', 'to', 'want', 'what', 'wher', 'you']


class ModelTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.intents = user_intents.Intents('intents1.json')
        cls.dataset = trainingdataset.TrainingDataset(cls.intents)
        cls.x_train = cls.dataset.x_train
        cls.y_train = cls.dataset.y_train

    # def test_train_model(self):
    #     model = models.TfModel(self.x_train, self.y_train)
    #     model.fit(self.x_train, self.y_train)
    #     import os.path
    #     self.assertTrue(os.path.isfile("model.tflearn.meta"))

    def test_train_torch_model(self):
        model = models.TorchModel(self.x_train, self.y_train)
        model.fit(self.x_train, self.y_train)
        import os.path
        self.assertTrue(os.path.isfile("model.pytorch"))



if __name__ == '__main__':
    unittest.main()
