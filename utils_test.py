import unittest

import dataset
import user_intents


class IntentsTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.intents = user_intents.Intents()
        cls.data = dataset.Dataset(cls.intents)
        cls.maxDiff = None

    def test_create_intent(self):
        self.assertIsNotNone(self.intents)

    def test_lexicon(self):
        self.assertEqual(self.intents.lexicon, lexicon)

    def test_classes(self):
        self.assertEqual(self.intents.classes, classes)

    def test_documents(self):
        self.assertEqual(self.intents.documents, documents)

    def test_bag_of_words(self):
        bag = dataset.build_bag_of_words(['is', 'anyon', 'ther', '?'], lexicon)
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


lexicon = ["'m", "'s", 'a', 'anyon', 'ar', 'buy', 'bye', 'can', 'cheap', 'cheapest', 'coupon', 'day', 'deal', 'find',
           'for', 'good', 'goodby', 'hello', 'help', 'hi', 'how', 'i', 'is', 'lat', 'less', 'look', 'me', 'med',
           'money', 'see', 'send', 'thank', 'that', 'the', 'ther', 'to', 'want', 'what', 'wher', 'you']

classes = ['coupon', 'goodbye', 'greeting', 'handleRx', 'med', 'thanks']

documents = [(['hi'], 'greeting'),
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

feature1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0]

feature5 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0]

label1 = [0, 0, 1, 0, 0, 0]

label5 = [0, 1, 0, 0, 0, 0]

if __name__ == '__main__':
    unittest.main()
