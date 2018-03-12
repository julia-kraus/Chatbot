import unittest

import chatbot


class ChatbotTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = chatbot.Chatbot('intents_med.json')

    def test_classify(self):
        result = self.bot.classify('Hello!')
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
