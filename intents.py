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
    intents_dict = {}

    def __init__(self, filename='intents_contextual_chatbot.json'):
        self.filename = filename
        intents = self.load_intents(filename)
        self.create_intents_dict(intents)
        self.organize_intents()

    def create_intents_dict(self, raw_intents):
        # for intent in raw_intents:
        #     self.intents[intent['tag']] = intent
        classes = [value['tag'] for value in raw_intents]
        self.intents_dict = {key: value for (key, value) in zip(classes, raw_intents)}

    @classmethod
    def load_intents(cls, filename):
        with open(filename) as json_data:
            intents = json.load(json_data)['intents']
        return intents

    def organize_intents(self):
        """extract words, documents and classes from the intents"""
        self.classes = word_utils.remove_duplicates(self.intents_dict.keys())
        for intent in self.intents_dict.values():
            for pattern in intent['patterns']:
                self.organize_patterns(pattern, intent)
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

    def get_document(self, pattern, intent):
        """get the documents from the intents"""
        w = word_utils.tokenize_sentence(pattern)
        self.documents.append({'words': w, 'class': intent['tag']})
        return

