import word_utils


class UserInput:
    """Class for the user input that is to be classified by the model"""

    def __init__(self, sentence):
        self.sentence = sentence
        self.words = self.get_words()

    def get_words(self):
        words = word_utils.tokenize_sentence(self.sentence)
        words = word_utils.stem(words)
        return words
