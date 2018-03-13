import unittest

import intents
import trainingdataset
import word_utils


class IntentsTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.int = intents.Intents('intents_med.json')
        cls.data = trainingdataset.TrainingDataset(cls.int)
        cls.maxDiff = None

    def test_create_intent(self):
        self.assertIsNotNone(self.int.intents)
        self.assertTrue('Hello, thanks for visiting' in self.int.intents['greeting']['responses'])

    def test_lexicon(self):
        self.assertEqual(self.int.lexicon, lexicon)

    def test_classes(self):
        self.assertEqual(self.int.classes, classes)

    # def test_documents(self):
    #     self.assertEqual(self.int.documents, documents)

    def test_responses(self):
        self.assertTrue(responses['greeting'][0] in self.int.intents['greeting']['responses'])

    def test_get_context(self):
        self.assertEqual(self.int.intents['med']['context_set'], 'getRx')

    def test_bag_of_words(self):
        bag = word_utils.build_bag_of_words(['is', 'anyon', 'ther', '?'], lexicon)
        self.assertEqual(bag, feature1)

    def test_get_features(self):
        features = self.data.get_features()
        self.assertEqual(features.shape, (len(documents), len(lexicon)))

    def test_get_labels(self):
        labels = self.data.get_labels()
        self.assertTrue((labels[1] == label1).all())
        self.assertTrue((labels[5] == label5).all())

    def test_shuffle_training_data(self):
        train_x, train_y = self.data.shuffle([feature1, feature5], [label1, label5])
        self.assertEqual(train_x.shape, (2, len(lexicon)))
        self.assertEqual(train_y.shape, (2, len(classes)))

    def test_get_training_data(self):
        train_x, train_y = self.data.get_training_data_from_intents()
        self.assertEqual(train_x.shape, (len(documents), len(lexicon)))
        self.assertEqual(train_y.shape, (len(documents), len(classes)))

    def test_save_training_data(self):
        self.data.save_data()
        import os.path
        self.assertTrue(os.path.isfile("training_data"))

    def test_load_training_data(self):
        lex, clss, x_train, y_train = self.data.load_data()
        self.assertEqual(lex, lexicon)
        self.assertEqual(clss, classes)
        self.assertEqual(len(x_train), len(documents))
        self.assertEqual(len(y_train), len(documents))


lexicon = ["'m", "'s", 'a', 'anyon', 'ar', 'buy', 'bye', 'can', 'cheap', 'cheapest', 'coupon', 'day', 'deal', 'find',
           'for', 'good', 'goodby', 'hello', 'help', 'hi', 'how', 'i', 'is', 'lat', 'less', 'look', 'me', 'med',
           'money', 'see', 'send', 'thank', 'that', 'the', 'ther', 'to', 'want', 'what', 'wher', 'you']

classes = ['coupon', 'goodbye', 'greeting', 'handleRx', 'med', 'thanks']

documents = [{'words': ['hi'], 'class': 'greeting'},
             (['how', 'ar', 'you'], 'greeting'),
             (['is', 'anyon', 'ther'], 'greeting'),
             (['hello'], 'greeting'),
             (['good', 'day'], 'greeting'),
             (['bye'], 'goodbye'),
             (['see', 'you', 'lat'], 'goodbye'),
             (['goodby'], 'goodbye'),
             (['thank'], 'thanks'),
             (['thank', 'you'], 'thanks'),
             (['that', "'s", 'help'], 'thanks'),
             (['i', "'m", 'look', 'for', 'cheap', 'med'], 'med'),
             (['want', 'to', 'find', 'a', 'deal'], 'med'),
             (['wher', 'ar', 'the', 'cheapest', 'med'], 'med'),
             (['wher', 'can', 'i', 'buy', 'med', 'for', 'less', 'money'], 'med'),
             (['what', 'is', 'the', 'coupon'], 'coupon'),
             (['send', 'me', 'the', 'coupon'], 'coupon')]

responses = {'greeting': ["Hello, thanks for visiting", "Good to see you again", "Hi there"],
             'goodbye': ["See you later", "Have a nice day", "Bye! Come back again soon."],
             'thanks': ["Happy to help!", "Any time!", "My pleasure"],
             'handleRx': [],
             'coupon': ["Sorry, no coupon", "No coupon available"]}

feature1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0]

feature5 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0]

label1 = [0, 0, 1, 0, 0, 0]

label5 = [0, 1, 0, 0, 0, 0]


class IntentsTester2():
    @classmethod
    def setUpClass(cls):
        cls.int = intents.Intents('intents_contextual_chatbot.json')
        cls.data = trainingdataset.TrainingDataset(cls.int)
        cls.maxDiff = None


if __name__ == '__main__':
    unittest.main()
