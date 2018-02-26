import nltk
import json

class Intents():
    """Class for Chatbot Intents. An intent is the user's intention when he interacts
    with the chatbot. 
    
    Attributes:
    words: all unique words contained in the documents
    documents: examples of user intents we have
    classes: possible types of intent (e.g. greeting, request, etc.)
    """
    intents = {}
    words = []
    classes = []
    documents = []
    
    def __init__(self, filename='intents1.json'):
        self.intents = self.load_intents(filename)
        self.words, self.classes, self.documents = self.organize_intents()
                 
    def load_intents(self, filename):
        """import our chatbots intent file"""
        with open('intents1.json') as json_data:
            intents = json.load(json_data)['intents']
        return intents
                 
    def organize_intents(self):
        """extract words, documents and classes from the intents"""
        for intent in self.intents:
            for pattern in intent['patterns']:
                self.organize_patterns(pattern)
        return 
    
    def organize_patterns(self, pattern):
            w = nltk.word_tokenize(pattern)
            self.words.extend(w)
  
            self.documents.append((w, intent['tag']))

            if intent['tag'] not in classes:
                self.classes.append(intent['tag'])
            return
            
    
def remove_duplicates(ls):
    return sorted(list(set(ls)))
                 
    