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
     document = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", document)
     lowers = document.lower()
     #remove the punctuation using the character deletion step of translate
     no_punctuation = re.sub(r'[^\w\s]','',lowers.strip())
     if len(document) > 0:
         return no_punctuation

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

text = train_df["text"]
test_text = test_df["text"]

docs = [clean_docs(i) for i in text]
test_docs = [clean_docs(i) for i in test_text]

'''
with open("clean.csv",'wb') as cleanFile:
    wr = csv.writer(cleanFile, dialect='excel')
    wr.writerow(docs)
    '''
count = Counter(tokens)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(docs)
weights = np.asarray(tfs.mean(axis=0)).ravel().tolist()

tfidf1000 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 1000)
tfs1000 = tfidf1000.fit_transform(docs)
idf = tfidf1000.idf_

tfidf500 = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 500)
tfs500 = tfidf500.fit_transform(docs)
features_list = tfidf500.get_feature_names()

doc

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb

tf_sublin = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features = 500, sublinear_tf = True)
tfs_sub = tf_sublin.fit_transform(docs)

lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, tfs500, train_df['age'], cv=4)
#[0.21032636, 0.19145488, 0.17784055, 0.20194009]

xgbm = xgb.XGBRegressor(n_estimators=360, max_depth=4, learning_rate=0.1)
cross_val_score(xgbm, tfs500, train_df['age'], cv=4)
#[0.22584977, 0.20422565, 0.19909738, 0.21166934]

lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, tfs_sub, train_df['age'], cv=4)
#[0.214408  , 0.19540301, 0.1822527 , 0.20712219]

xgbm = xgb.XGBRegressor(n_estimators=360, max_depth=4, learning_rate=0.1)
cross_val_score(xgbm, tfs_sub, train_df['age'], cv=4)
#[0.22664647, 0.20397692, 0.19850794, 0.21042486]

test_data = tfidf500.transform(test_docs)

lasso.fit(tfs500, train_df['age'])
preds_lasso = lasso.predict(test_data)
solution_lasso = pd.DataFrame({"user.id":test_df["user.id"], "age":preds_lasso})
solution_lasso = solution_lasso.groupby('user.id', as_index=False)['age'].mean()
solution_lasso.to_csv("solutions/word_vecs_metadata_lasso_normalization.csv", index = False)

pd.DataFrame(tfs500.toarray(),columns=features_list).to_csv("train_tf-idf.csv")
pd.DataFrame(test_data.toarray(),columns=features_list).to_csv("test_tf-idf.csv")
