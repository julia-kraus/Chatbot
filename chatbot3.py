
                                         
    def create_training_data():
            # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            # list is not necessary because output_empty is already a list
            # index function: 
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        return train_x, train_y
                 

def remove_duplicates(ls):
    return sorted(list(set(ls)))

def stem_words(words):
    ignore_words = ['?']
    # stem and lower each word and remove duplicates. set removes duplicates, then turn back to list
    return [stemmer.stem(w.lower()) for w in words if w not in ignore_words]

def create_model():
                 
       
