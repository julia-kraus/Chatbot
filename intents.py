import json

import word_utils


class Intents:
    """Class for chatbot Intents. An intent is the user's intention when he interacts
    with the chatbot. 
    
    Attributes:
    words: all unique words contained in the documents
    documents: examples of user intents we have
    classes: possible types of intent (e.g. greeting, request, etc.)
    """

    lexicon = []
    classes = []
    documents = []
    # dictionary of responses
    responses = {}

    def __init__(self, filename='intents_med.json'):
        self.intents = self.load_intents(filename)
        self.organize_intents()

    @classmethod
    def load_intents(cls, filename):
        with open(filename) as json_data:
            intents = json.load(json_data)['intents']
        return intents

    def organize_intents(self):
        """extract words, documents and classes from the intents"""
        for intent in self.intents:
            self.get_class(intent)
            self.get_responses(intent)
            for pattern in intent['patterns']:
                self.organize_patterns(pattern, intent)

        self.classes = word_utils.remove_duplicates(self.classes)
        self.lexicon = word_utils.remove_duplicates(self.lexicon)
        return

    def organize_patterns(self, pattern, intent):
        """organizes the lexicon, class and documents"""
        self.get_lexicon(pattern)
        self.get_document(pattern, intent)
        return

    def get_lexicon(self, pattern):
        """tokenizes a pattern and adds the resulting words to the lexicon"""
        w = word_utils.tokenize_sentence(pattern)
        self.lexicon.extend(w)
        return

    def get_class(self, intent):
        """get the unique classes of the intents"""
        if intent['tag'] not in self.classes:
            self.classes.append(intent['tag'])
        return

    def get_document(self, pattern, intent):
        """get the documents from the intents"""
        w = word_utils.tokenize_sentence(pattern)
        self.documents.append((w, intent['tag']))
        return

    def get_responses(self, intent):
        tag = intent['tag']
        self.responses[tag] = intent['responses']
