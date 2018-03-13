import unittest

import chatbot


class ChatbotTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = chatbot.Chatbot('intents_med.json')

    def test_classify(self):
        result = self.bot.classify('Hello!')
        self.assertTrue("greeting" in result[0])

    def test_response(self):
        self.answer = self.bot.respond('Hello!')
        self.assertTrue(self.answer in ["Hello, thanks for visiting", "Good to see you again", "Hi there"])




if __name__ == '__main__':
    unittest.main()
