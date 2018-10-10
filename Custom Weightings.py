import numpy as np
#nltk.download('punkt')
import re
#nltk.download('stopwords')
import  pandas as pd
import math
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

#implements various methods on text to clean
def clean_docs(document):
    
     document = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", document) #remove numbers
     lowers = document.lower()
     
     #remove the punctuation using regukar expression
     no_punctuation = re.sub(r'[^\w\s]','',lowers.strip())
     if len(document) > 0:
         return no_punctuation
    
#this function calculates probability idf
def probidf(docfreq, totaldocs, log_base=2.0, add=0.0):
    return add + math.log(1.0 * (totaldocs - docfreq) / docfreq, log_base)
     
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

text = train_df["text"]
test_text = test_df["text"]
train_catch = text.shape[0]

all_text = pd.concat((text, test_text)).reset_index(drop=True) #if combining test and train
test_user = test_df['user.id']

#different loadable lists of words to use as dictionary
import pickle
with open("featurelist.txt", "rb") as fp:   # Unpickling feature list
     features_list = pickle.load(fp)
     
with open("wholemodelfeaturelist.txt", "rb") as fp: 
    whole_features = pickle.load(fp)

with open("ngramfeaturelist.txt", "rb") as fp:   #Pickling
    whole_features = pickle.load(fp)
    
#cleaning train and test
docs = [clean_docs(i) for i in text]
test_docs = [clean_docs(i) for i in test_text]

#creating a dictionary for the top 200 most occuring words 
split = [features_list]
dct = Dictionary(split)

#creating a dictionary for the top 200 most occuring words, all text model
docs = [clean_docs(i) for i in all_text]
split = [whole_features]
dct = Dictionary(split)

#splitting corpus and getting word counts for words in the dictionary
document_list = [i.split() for i in docs]
raw_corpus = [dct.doc2bow(t) for t in document_list]

### Generating different tf-idf models

#regular idf
tf_200_mod = TfidfModel(corpus = raw_corpus)
tfidf_200 = tf_200_mod[raw_corpus]

#probability idf
tf_prob_mod = TfidfModel(corpus = raw_corpus, wglobal = probidf)
prob_tfidf = tf_prob_mod[raw_corpus]
y = prob_tfidf[442960]

#log plus 1 tf, probability idf
tf_weight_3 = TfidfModel(corpus = raw_corpus, wlocal = np.log1p, wglobal = probidf)
weight3_tfidf = tf_weight_3[raw_corpus]

#smooth tf
tf_log_norm = TfidfModel(corpus = raw_corpus, wlocal = np.log1p)
logNorm_tfidf = tf_log_norm[raw_corpus]

#boolean tf, probability idf, cosine normalization
tf_bool_mod = TfidfModel(corpus = raw_corpus, smartirs = 'bpc')
tfidf_bool = tf_bool_mod[raw_corpus]

#turning training data into a pdf, not in full text model, tf-idf changed for different models
train_200 = pd.DataFrame.from_records([{v: k for v, k in row} for row in tfidf_bool])
train_200.fillna(0, inplace = True)

#this process drops some columns from the list of features so a new setup is needed to name cols
cols = list(train_200.columns.values)
train_200.columns = [features_list[i] for i in cols]

#transforming the test data, if not in informed model
document_list = [i.split() for i in test_docs]
test_corpus = [dct.doc2bow(t) for t in document_list]
test_200 = tf_bool_mod[test_corpus]
test_200[14444]
test_200 = pd.DataFrame.from_records([{v: k for v, k in row} for row in test_200])
test_200.fillna(0, inplace = True)
cols = list(test_200.columns.values)
test_200.columns = [features_list[i] for i in cols]

#for a combined model doing the same then splitting
_200 = pd.DataFrame.from_records([{v: k for v, k in row} for row in tfidf_bool])
_200.fillna(0, inplace = True)
cols = list(_200.columns.values)
_200.columns = [whole_features[i] for i in cols]
train_200 = _200[:train_catch]
test_200 = _200[train_catch:]
test_200 = test_200.reset_index(drop = True)

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.model_selection import cross_val_score

#testing the model, labeled cv for each idf above full indicates train and test were combined
lasso = Lasso(alpha =0.0005, random_state=1)
cross_val_score(lasso, train_200, train_df['age'], cv=4)
#[0.14402136, 0.11481024, 0.11340008, 0.11876017] 200 features
#[0.14387297, 0.11472049, 0.11332101, 0.11870199] 200 For full model
#[0.13967703, 0.11201649, 0.11076932, 0.11657563] For probability idf
#[0.13962651, 0.11042061, 0.1117899 , 0.11712385] log_norm tf, prob idf
#[0.14476047, 0.11379   , 0.11520098, 0.11985188] log norm tf
#[0.13051224, 0.10063653, 0.10559687, 0.10833461] bool tf, prob idf, cos normalization
#[0.13033494, 0.10058716, 0.1055756 , 0.10831238] bool prob cos full model
#[0.13952379, 0.11194307, 0.11072421, 0.11654045] natural prob cos full

### The best model used a boolean tf, probability idf, and cosine normalization

#grouping by user id
train_200['user.id'] = train_df['user.id']
train_200 = train_200.groupby('user.id', as_index=False).mean()

test_200['user.id'] = test_user
test_200 = test_200.groupby('user.id', as_index=False).mean()

#if not using combined model, concatinating to make up for uneven columns before splitting
tt= pd.concat((train_200, test_200)).reset_index(drop=True)
tt.fillna(0, inplace = True)
train_200 = tt[:train_catch]
test_200 = tt[train_catch:] 

#saving results as csv
test_200.to_csv("bool_test_tf-idf.csv")
train_200.to_csv("bool_train_tf-idf.csv")

