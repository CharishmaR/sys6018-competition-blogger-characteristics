import numpy as np
import nltk
#nltk.download('punkt')
import re
#nltk.download('stopwords')
import  pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

#uses nltk to stem a set of tokens according to PorterStemmer()
stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#uses nltk to tokenize a piece of text and implement stem
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#implements various methods on text to clean
def clean_docs(document):
    
     document = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", document) #remove numbers
     lowers = document.lower()
     
     #remove the punctuation using regukar expression
     no_punctuation = re.sub(r'[^\w\s]','',lowers.strip())
     if len(document) > 0:
         return no_punctuation

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

text = train_df["text"]
test_text = test_df["text"]

#if combining test and train
train_catch = text.shape[0]
text = pd.concat((text, test_text)).reset_index(drop=True)

#cleaning documents
docs = [clean_docs(i) for i in text]
test_docs = [clean_docs(i) for i in test_text]

'''
with open("clean.csv",'wb') as cleanFile:
    wr = csv.writer(cleanFile, dialect='excel')
    wr.writerow(docs)
    '''

#full tf idf
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(docs)
weights = np.asarray(tfs.mean(axis=0)).ravel().tolist()

#1000 feature tf idf
tfidf1000 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 1000)
tfs1000 = tfidf1000.fit_transform(docs)
idf = tfidf1000.idf_

#500 feature tf idf
tfidf500 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 500)
tfs500 = tfidf500.fit_transform(docs)
features_list = tfidf500.get_feature_names()

#100 feature tf idf
tfidf100 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 100)
tfs100 = tfidf100.fit_transform(docs)
features_list = tfidf100.get_feature_names()

#200 feature tf idf
tfidf200 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 200)
tfs200 = tfidf200.fit_transform(docs)
features_list = tfidf200.get_feature_names()

#500 feature tfidf with ngram = 2
tfidf_ngram = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 500, ngram_range=(1,2))
tfs_ngram = tfidf_ngram.fit_transform(docs)
features_list = tfidf_ngram.get_feature_names()

#sublinear tfidf
tf_sublin = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 500, sublinear_tf = True)
tfs_sub = tf_sublin.fit_transform(docs)

#dumping feature lists to different files to use as dictionaries
import pickle
with open("ngramfeaturelist.txt", "wb") as fp:   #Pickling
    pickle.dump(features_list, fp)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score

#if combining train and test
train_tf = tfs_ngram[:train_catch]
test_tf = tfs_ngram[train_catch:] 

#testing lasso 500 features
lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, train_tf, train_df['age'], cv=4)
#[0.21032636, 0.19145488, 0.17784055, 0.20194009] regular 500
#[0.20870566, 0.19208768, 0.17827907, 0.2014498] ngram 500

#testing lasso sublinear, 500 features
lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, tfs_sub, train_df['age'], cv=4)
#[0.214408  , 0.19540301, 0.1822527 , 0.20712219]

#testing lassos for 100 features
lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, tfs100, train_df['age'], cv=4)
#[0.1380533 , 0.11513725, 0.11607548, 0.11817014]

#testing lasso for 200 features
lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, train_tf, train_df['age'], cv=4)
#[0.16728556, 0.14460069, 0.14029157, 0.15055411]
#[0.1672003 , 0.14455272, 0.1402375 , 0.15052539] Full

#saving csv
pd.DataFrame(tfs500.toarray(),columns=features_list).to_csv("train_tf-idf.csv")
pd.DataFrame(test_data.toarray(),columns=features_list).to_csv("test_tf-idf.csv")
