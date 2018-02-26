    def build_bag_of_words(intents):
        """return a bag of words for each sentence"""
            for doc in intents.documents:
                bag = []
                words = get_stemmed_words_from_document(doc)

                # create our bag of words array
                for w in intents.lexicon:
                    bag.append(1) if w in words else bag.append(0)

                output_row[classes.index(doc[1])] = 1
                training_data.append([bag, output_row])
                
    def get_labels(intents):
    for doc in intents.documents:
        label = [classes.index(doc[1])] = 1
                
def create_training_data(intents):
    
    training_features = build_bag_of_word(intents)
    training_labels = get_labels(intents)
    
    training_data = list(zip(training_features, training_labels))

    random.shuffle(training)
    
    training_x, training_y = zip(*training_data)
    
    return train_x, train_y


            
            
def get_words_from_document(doc):
    words = doc[0]

def create_training_feature():
    pass

def create_training_label():
    pass
    