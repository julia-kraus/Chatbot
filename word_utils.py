import nltk
from nltk.stem import LancasterStemmer


# from nltk.stem.snowball import GermanStemmer


def stem(words):
    stemmer = LancasterStemmer()
    ignore_words = ['?']
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    return words


def remove_duplicates(ls):
    return sorted(list(set(ls)))


def tokenize_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = stem(words)
    return words


def build_bag_of_words(words, lexicon):
    """input: lexicon and list of words. Return: bag of words"""
    bag = []
    for w in lexicon:
        bag.append(1) if w in words else bag.append(0)
    return bag
