import numpy as np
import nltk
#nltk.download('punkt')
import re
#nltk.download('stopwords')
import  pandas as pd
import csv

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def clean_docs(document):
     lowers = document.lower()
     #remove the punctuation using the character deletion step of translate
     no_punctuation = re.sub(r'[^\w\s]','',lowers.strip())
     if len(document) > 0:
         return no_punctuation

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

text = train_df["text"]

docs = [clean_docs(i) for i in text]
with open("clean.csv",'wb') as cleanFile:
    wr = csv.writer(cleanFile, dialect='excel')
    wr.writerow(docs)
    
count = Counter(tokens)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(docs)
weights = np.asarray(tfs.mean(axis=0)).ravel().tolist()

tfidf1000 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 1000)
tfs1000 = tfidf1000.fit_transform(docs)
features = tfidf1000.get_feature_names()
idf = tfidf1000.idf_

stuff = pd.DataFrame(tfs1000.toarray(), 
columns=features)
stuff.to_csv("tf-idf.csv")

###To Do: 1. Remove numbers, look at other idf possibilities, work on a test set, try to predict age
