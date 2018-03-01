import unittest
import utils


class IntentsTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.intents = utils.Intents()
        cls.dataset = utils.Dataset()
        cls.maxDiff = None

    def test_create_intent(self):
        self.assertIsNotNone(self.intents)

    def test_lexicon(self):
        self.assertEqual(self.intents.lexicon, lexicon)

    def test_classes(self):
        self.assertEqual(self.intents.classes, classes)

    def test_documents(self):
        self.assertEqual(self.intents.documents, documents)

    def test_create_training_data(self):
        pass


lexicon = ["'m", "'s", 'a', 'anyon', 'ar', 'buy', 'bye', 'can', 'cheap', 'cheapest', 'coupon', 'day', 'deal', 'find',
           'for',
           'good', 'goodby', 'hello', 'help', 'hi', 'how', 'i', 'is', 'lat', 'less', 'look', 'me', 'med', 'money',
           'see',
           'send', 'thank', 'that', 'the', 'ther', 'to', 'want', 'what', 'wher', 'you']

classes = ['coupon', 'goodbye', 'greeting', 'med', 'thanks']

documents = [(['Hi'], 'greeting'), (['How', 'are', 'you'], 'greeting'), (['Is', 'anyone', 'there', '?'], 'greeting'),
             (['Hello'], 'greeting'), (['Good', 'day'], 'greeting'), (['Bye'], 'goodbye'),
             (['See', 'you', 'later'], 'goodbye'),
             (['Goodbye'], 'goodbye'), (['Thanks'], 'thanks'), (['Thank', 'you'], 'thanks'),
             (['That', "'s", 'helpful'], 'thanks'),
             (['I', "'m", 'looking', 'for', 'cheap', 'meds'], 'med'), (['want', 'to', 'find', 'a', 'deal'], 'med'),
             (['where', 'are', 'the', 'cheapest', 'meds'], 'med'),
             (['where', 'can', 'I', 'buy', 'meds', 'for', 'less', 'money'], 'med'),
             (['what', 'is', 'the', 'coupon'], 'coupon'), (['send', 'me', 'the', 'coupon'], 'coupon')]

if __name__ == '__main__':
    unittest.main()
