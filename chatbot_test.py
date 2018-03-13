import unittest

import chatbot


class ChatbotTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = chatbot.Chatbot('intents_contextual_chatbot.json')

    def test_classify(self):
        result = self.bot.classify('Hello!')
        self.assertTrue("greeting" in result[0])

    def test_response(self):
        self.answer = self.bot.respond('Hello!')
        self.assertTrue(self.answer in ["Hello, thanks for visiting", "Good to see you again", "Hi there"])

    def contextPresponse(self):
        self.answer1 = self.bot.respond('Can I rent a moped?')
        self.assertTrue(self.answer1 in ["Are you looking to rent today or later this week?"])
        self.answer2 = self.bot.respond('today')
        self.assertTrue(self.answer2 in ["For rentals today please call 1-800-MYMOPED",
                                         "Same-day rentals please call 1-800-MYMOPED"])


if __name__ == '__main__':
    unittest.main()
