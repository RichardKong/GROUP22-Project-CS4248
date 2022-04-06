import pandas as pd
import spacy
import re

nlp = spacy.load('en_core_web_sm')

def preprocess(dataframe):
    length = len(dataframe)
    feat1 = [""] * length
    feat2 = [""] * length
    for i in range(length):
        b = dataframe.iloc[i]
        feat1[i] = count_ner(b["text"])
        feat2[i] = count_number_of_pronoun(b["text"])
    dataframe["ner_count"] = feat1
    dataframe["pronoun_count"] = feat2
    return dataframe    

def abbreviation_replacement(text):
    text = re.sub(r"nt", "not", text)
    text = re.sub(r"im", "i am", text)
    # text = re.sub(r"\'re", "are", text)
    text = re.sub(r"hes", "he is", text)
    text = re.sub(r"she\'s", "he is", text)
    # text = re.sub(r"it\'s", "it is", text)
    # text = re.sub(r"that\'s", "that is", text)
    # text = re.sub(r"who\'s", "who is", text)
    # text = re.sub(r"what\'s", "what is", text)
    # text = re.sub(r"n\'t", "not", text)
    # text = re.sub(r"\'ve", "have", text)
    # text = re.sub(r"\'d", "would", text)
    # text = re.sub(r"\'ll", "will", text)
    # text = re.sub(r",", " , ", text)
    # text = re.sub(r"!", " ! ", text)
    # text = re.sub(r"\.", " \. ", text)
    # text = re.sub(r"\(", " \( ", text)
    # text = re.sub(r"\)", " \) ", text)
    # text = re.sub(r"\?", " \? ", text)
    return text

def count_ner(message):
    doc = nlp(message)
    return len(doc)

def count_number_of_pronoun(message):
    pronoun = ("i", "we", "me", "you", "they", "he", "she", "me", "him", "them", "her")
    return sum([1 for x in message.split() if x.lower() in pronoun])


def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train_1 = preprocess(train)
    test_1 = preprocess(test)
    train_1.to_csv("train_extended.csv")
    test_1.to_csv("test_extended.csv")


if __name__ == "__main__":
    main()