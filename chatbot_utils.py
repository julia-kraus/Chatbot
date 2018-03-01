import nltk
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

class Intents():
    """Class for Chatbot Intents. An intent is the user's intention when he interacts
    with the chatbot. 
    
    Attributes:
    words: all unique words contained in the documents
    documents: examples of user intents we have
    classes: possible types of intent (e.g. greeting, request, etc.)
    """
    
    lexicon = []
    classes = []
    documents = []
    
    def __init__(self, filename='intents1.json'):
        self.intents = self.load_intents(filename)
        self.organize_intents()
    
                 
    def load_intents(self, filename):
        """import our chatbots intent file"""
        with open('intents1.json') as json_data:
            intents = json.load(json_data)['intents']
        return intents
                 
    def organize_intents(self):
        """extract words, documents and classes from the intents"""
        for intent in self.intents:
            for pattern in intent['patterns']:
                self.organize_patterns(pattern, intent)
                
        self.edit_lexicon()
        self.classes = remove_duplicates(self.classes)
        return
    
    def organize_patterns(self, pattern, intent):
        """organizes the lexicon, pattern and documents"""
        self.get_lexicon(pattern)
        self.get_class(intent)
        self.get_document(pattern, intent)
        return 
        
    def get_lexicon(self, pattern):
        """tokenizes a pattern and adds the resulting words to the lexicon"""
        w = nltk.word_tokenize(pattern)
        self.lexicon.extend(w)
        return
    
    def get_class(self, intent):
        """get the unique classes of the intents"""
        if intent['tag'] not in self.classes:
             self.classes.append(intent['tag'])
        return
    
    def get_document(self, pattern, intent):
        """get the documents from the intents"""
        w = nltk.word_tokenize(pattern)
        self.documents.append((w, intent['tag']))
        return
    
    def edit_lexicon(self):
        """remove duplicates and stem words"""
        self.lexicon = stem(self.lexicon)
        self.lexicon = remove_duplicates(self.lexicon)
        return
    
/* class Dataset():
    """Class for creation of training- and test datasets."""
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    
    def __init__(train_intents, test_intents = None):
        self.x_train = self.create_features(train_intents)
        self.x_test = self.create_features(test_intents)
        self.y_train = self.create_features
    
    def create_features(intents):
        training_features = []
        for doc in intents.documents:
            bag = intents.build_bag_of_words(doc)
            training_features.append(bag)
        return training_features
            
        
    def build_bag_of_words(self, doc):
        """return a bag of words a sentence"""
        bag = []
        words = get_words_from_document(doc)
        words = stem(words)

        for w in self.lexicon:
            bag.append(1) if w in words else bag.append(0)
        return bag
                
    def get_labels(self):
        """stimmt noch nicht ganz"""
        labels = []
        for doc in self.documents:
            label = [classes.index(doc[1])] == 1
            labels.append(label)
        return labels
                
    def create_training_data(self):
    
        training_features = self.create_features
        training_labels = self.get_labels()
    
        training_data = list(zip(training_features, training_labels))

        random.shuffle(training)
    
        return zip(*training_data)
    
*/ 

def get_words_from_document(doc):
    words = doc[0]
    return words
   
def stem(words):
    ignore_words = ['?']
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    return words

    
def remove_duplicates(ls):
    return sorted(list(set(ls)))

def classify():
    pass

def response():
    pass


    
                 

                 
    