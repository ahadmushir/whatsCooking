#using xgboost

import pandas as pd
import xgboost as xgb
import difflib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt  
from fuzzywuzzy import fuzz
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
train_df = pd.read_csv('newXgTrain.csv', header=0)
test_df = pd.read_csv('newXgTest.csv', header=0)

# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

#feature handling

# a = train_df.columns

# feature_columns_to_use = list()


# a1 = a[2:]

# la = list()
# ov = list()

# for i in a1:
# 	for j in a1:
# 		d = fuzz.token_sort_ratio(i,j)
# 		if d != 100 and d > 83:
# 			print i,j
			
# 			if i not in la and j not in la:
# 				la.append([i,j])

# for i in la:
# 	for j in i:
# 		if j not in ov:
# 			ov.append(j)

# train_df['new'] = train_df[la[0][0]] | train_df[la[0][1]]
# train_df['new1'] = train_df[la[1][0]] | train_df[la[1][1]]
# train_df['new2'] = train_df[la[2][0]] | train_df[la[2][1]]
# train_df['new3'] = train_df[la[3][0]] | train_df[la[3][1]]
# train_df['new4'] = train_df[la[4][0]] | train_df[la[4][1]]
# train_df['new5'] = train_df[la[5][0]] | train_df[la[5][1]]
# train_df['new6'] = train_df[la[6][0]] | train_df[la[6][1]]
# train_df['new7'] = train_df[la[7][0]] | train_df[la[7][1]]
# train_df['new8'] = train_df[la[8][0]] | train_df[la[8][1]]
# train_df['new9'] = train_df[la[9][0]] | train_df[la[9][1]]
# train_df['new10'] = train_df[la[10][0]] | train_df[la[10][1]]
# train_df['new11'] = train_df[la[11][0]] | train_df[la[11][1]]
# train_df['new12'] = train_df[la[12][0]] | train_df[la[12][1]]
# train_df['new13'] = train_df[la[13][0]] | train_df[la[13][1]]
# train_df['new14'] = train_df[la[14][0]] | train_df[la[14][1]]
# train_df['new15'] = train_df[la[15][0]] | train_df[la[15][1]]
# train_df['new16'] = train_df[la[16][0]] | train_df[la[16][1]]
# train_df['new17'] = train_df[la[17][0]] | train_df[la[17][1]]
# train_df['new18'] = train_df[la[18][0]] | train_df[la[18][1]]
# train_df['new19'] = train_df[la[19][0]] | train_df[la[19][1]]

# v = a[321:]
# cc = 2
# while cc != len(train_df.columns):
# 	if a[cc] not in ov:

# 		feature_columns_to_use.append(a[cc])
# 	cc = cc + 1
# print len
# print b



feature_columns_to_use = list()
aData = train_df.columns 

a1 = aData[2:]

la = list()
ov = list()
####
#Trying to merge ingredients which are similar (without the use of lemmatizer)
for i in a1:
	for j in a1:
		d = fuzz.token_sort_ratio(i,j)
		if d != 100 and d > 75:
			print i,j
			
			if i not in la and j not in la:
				la.append([i,j])

for i in la:
	for j in i:
		if j not in ov:
			ov.append(j)



print 'len of not including ing',len(ov)
print 'initial cols', len(train_df.columns)

ran = range(len(ov))

####
##for train
cc = 0
while cc != len(ov):

	
	c1 = 0
	cStr = str(cc)

	train_df[cStr] = train_df[la[c1][0]] | train_df[la[c1][1]]
	cc = cc + 1
	c1 = c1 + 1

##for test
cc = 0
while cc != len(ov):

	
	c1 = 0
	cStr = str(cc)

	test_df[cStr] = test_df[la[c1][0]] | test_df[la[c1][1]]
	cc = cc + 1
	c1 = c1 + 1
print train_df.columns[len(ov)+1:len(train_df.columns)-1]

print 'aggregated columns',len(train_df.columns)

a = train_df.columns


ccc = 2
while ccc != len(a):
	if a[ccc] not in ov:

		feature_columns_to_use.append(a[ccc])
	ccc = ccc + 1

print 'final length of cols',len(feature_columns_to_use)


# print len(feature_columns_to_use)

# feature_columns_to_use = ['salt','onions','garlic','olive oil','sugar','water','soy sauce','carrots','butter','garlic cloves', 'ground black pepper', 'eggs', 'vegetable oil']

# feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
# nonnumeric_columns = ['Sex']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different

big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.

# le = LabelEncoder()
# for feature in nonnumeric_columns:
#     big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])


# Prepare the inputs for the model

train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['class']


gbm = xgb.XGBClassifier(max_depth=15, n_estimators=700, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)


# clf = xgb.XGBClassifier(n_estimators=200)
# eval_set  = [(train_X,train_y)]	
# clf.fit(train_X, train_y, eval_metric="auc")

# mapFeat = dict(zip(["f"+str(i) for i in range(len(feature_columns_to_use))],feature_columns_to_use))
# ts = pd.Series(gbm.booster().get_fscore())
# ts.index = ts.reset_index()['index'].map(mapFeat)
# ts.order()[-15:].plot(kind="barh", title=("features importance"))
# plt.show()
# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.

# corpustr = train_df[feature_columns_to_use]
# corr = train_df['class']

# vectorizertr = TfidfVectorizer(stop_words='english', ngram_range = ( 1, 1),analyzer="word", 
#                               max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False, norm = 'l2')

# tfidftr = vectorizertr.fit_transform(corpustr).todense()
# clss = vectorizertr.fit_transform(corr).todense()

# nb = BernoulliNB()



# nb = nb.fit(tfidftr,clss)
# scores = cross_validation.cross_val_score(nb, tfidftr, clss, cv=5, scoring='accuracy')
# print("Accuracy BernoulliNB: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))	




submission = pd.DataFrame({ 'cuisine': predictions,
							'id': test_df['id']	 })
submission.to_csv("submissionX14Dec.csv", index=False)
