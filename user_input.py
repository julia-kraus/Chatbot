import word_utils


class UserInput:
    """Class for the user input that is to be classified by the model
    Import and classify input, give response"""
    ERROR_TRESHOLD = 0.25
    context = {}

    def __init__(self, sentence, lexicon):
        self.sentence = sentence
        self.words = self.get_words()
        self.bag = word_utils.build_bag_of_words(self.words, lexicon)

    def get_words(self):
        words = word_utils.tokenize_sentence(self.sentence)
        words = word_utils.stem(words)
        return words
