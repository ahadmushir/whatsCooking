import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


"""
Soft Voting/Majority Rule classifier

This module contains a Soft Voting/Majority Rule classifier for
classification clfs.

"""
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
#Reading training file
traindf = pd.read_json("../input/train.json")

# traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]
####
#using lematizer from NLTK library 
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

#Reading Test
testdf = pd.read_json("../input/test.json") 
# testdf['ingredients_clean_string'] = [' , '.join(z).strip() for z in testdf['ingredients']]
####
#using lematizer from NLTK library 
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       

####
#Making TF-IDF vector
corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')
#vectorizertr = HashingVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 2 ),analyzer="word", token_pattern=r'\w+' , n_features = 7000)
                             
tfidftr = vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')

#vectorizerts = HashingVectorizer(stop_words='english',
#                             ngram_range = ( 1 , 2 ),analyzer="word", token_pattern=r'\w+' , n_features = 7000)

tfidfts = vectorizertr.transform(corpusts)

predictors_tr = tfidftr

targets_tr = traindf['cuisine']

predictors_ts = tfidfts

################################
# Initialize classifiers
################################

np.random.seed(1)
print("Ensemble: LR - linear SVC")
clf1 = LogisticRegression(random_state=1, C=7)
clf2 = LinearSVC(random_state=1, C=0.4, penalty="l2", dual=False)
nb = BernoulliNB()
rfc = RandomForestClassifier(random_state=1, criterion = 'gini', n_estimators=500)
sgd = SGDClassifier(random_state=1, alpha=0.00001, penalty='l2', n_iter=50)


eclf = EnsembleClassifier(clfs=[clf1, clf2,nb, rfc, sgd], weights=[2, 2, 1, 1,2])
np.random.seed(1)
for clf, label in zip([eclf], 
    ['Ensemble']):
    scores = cross_validation.cross_val_score(clf, predictors_tr,targets_tr, cv=2, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))



eclf2 = eclf.fit(predictors_tr,targets_tr)
predictions = eclf2.predict(predictors_ts)
testdf['cuisine'] = predictions
#testdf = testdf.sort('id' , ascending=True)
testdf = testdf.sort_values(by='id' , ascending=True)


##show the detail
# testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("info_vote2.csv")

#for submit, no index
testdf[['id' , 'cuisine' ]].to_csv("just_python_cooking-vote1.csv", index=False)
