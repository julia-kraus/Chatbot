ERROR_TRESHOLD = 0.25


def classify(sentence, model, classes):
    # generate probabilities from the model
    results = model.predict()
    # filter out predictions elow a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_TRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list
