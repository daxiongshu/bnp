import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble, preprocessing, decomposition, cross_validation, feature_selection
import xgboost as xgb
import sys

from sklearn.metrics import log_loss

def logloss(y_true, y_pred):
	y_pred = np.clip(y_pred, 1e-15, 1.-1e-15)
	return log_loss(y_true, y_pred)
	
def save(train, test, train_index, test_index, y):

	train['target'] = y
	train['ID'] = train_index
	test['ID'] = test_index
	train.set_index('ID', inplace=True)
	test.set_index('ID', inplace=True)

	train.to_csv("train.v2.csv")
	test.to_csv("test.v2.csv")
	
def evaluate(X, y):	
	scores = cross_validation.cross_val_score(LogisticRegression(), X, y, cv=kf, n_jobs=len(kf), scoring='log_loss')
	score = np.mean(scores)
	print scores, -score
	return -score

def evaluate2(X, y):
		  
	trscores, cvscores = [], []
	trpreds, trtrues = [], []
	
	for itr, icv in kf:

		X_train, X_valid    = X[itr], 	X[icv]
		y_train, y_valid    = y[itr],   y[icv]

		print X_train.shape, X_valid.shape

# 		clf = xgb.XGBClassifier()
		clf = LogisticRegression()
		clf.fit(X_train, y_train)
# 		clf.set_params(**pp)
# 		clf.fit(
# 			X_train, y_train, 
# 			eval_set=[(X_valid, y_valid)], 
# 			eval_metric="logloss",
# 			early_stopping_rounds = early_stopping_rounds,
# 		)

		trpred = clf.predict_proba(X_train)[:,1]
		trscore = logloss(y_train, trpred)
		cvpred = clf.predict_proba(X_valid)[:,1]
		cvscore = logloss(y_valid, cvpred)
			
		trpreds += list(cvpred)
		trtrues += list(y_valid)

		print trscore, cvscore
		
		trscores.append(trscore)
		cvscores.append(cvscore)

	score = np.mean(cvscores)

	print cvscores, score

	return score

def linear_select_interactions(train, y, n_interactions, test, train_index, test_index, scoring=logloss):
	"""Creates interactions and determines if they are useful by fitting a model and evaluating with CV"""

	from sklearn.cross_validation import cross_val_score
	from sklearn.linear_model import LogisticRegression
	
	for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
		if train_series.dtype == 'O' or test_series.dtype == 'O':
			train[train_name], tmp_indexer = pd.factorize(train[train_name])
			test[test_name] = tmp_indexer.get_indexer(test[test_name])

	print 'Generating and selecting useful feature interactions'
	
	train.fillna(-1, inplace=True)
	test.fillna(-1, inplace=True)
	
	categorical = list(train.columns[train.dtypes != np.float64])
	continues = list(train.columns[train.dtypes == np.float64])
	
	for col in []+categorical:
		if train[col].max() > 150:
			categorical.remove(col)
			continues.append(col)
	
	for col in train.columns:
		mean = train[col].mean()
		std = train[col].std(ddof=0)
		train[col] = (train[col] - mean) / std
		test[col] = (test[col] - mean) / std
	
	for col in categorical:
		a = pd.get_dummies(pd.concat((train[col].astype(str), test[col].astype(str))))
		train_a = a.head(train.shape[0])
		test_a = a.tail(test.shape[0])
		
		for ncol in train_a.columns:
			train[col+str(ncol)] = train_a[ncol]
			test[col+str(ncol)] = test_a[ncol]
		
	train_X = train.values
	test_X = test.values
	
	print train_X.shape
	
	save(train.copy(), test.copy(), train_index, test_index, y)

	# Get baseline performance of the model
	score = evaluate(train_X, y)
	best_score = score
	print best_score
	
	train_best = train.copy()
	test_best = test.copy()

	for col in categorical:
		for col2 in continues:
		
			train = train_best.copy()
			test = test_best.copy()
			
			a = pd.get_dummies(pd.concat((train[col].astype(str), test[col].astype(str))))
			train_a = a.head(train.shape[0])
			test_a = a.tail(test.shape[0])
		
			for ncol in train_a.columns:
				train[col+str(ncol)+"*"+col2] = train_a[ncol] * train[col2]
				test[col+str(ncol)+"*"+col2] = test_a[ncol] * test[col2]
				
			train_X = train.values
			test_X = test.values
			
			print (col, col2), train_X.shape, test_X.shape

			score = evaluate(train_X, y)
			
			if score < best_score:
				print "better", score
				best_score = score
				train_best = train.copy()
				test_best = test.copy()
				
				save(train.copy(), test.copy(), train_index, test_index, y)
				
	return train, test
			

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

y = target = train['target'].values

kf = cross_validation.StratifiedKFold(y, 3, shuffle=True, random_state=11)

test_index = test['ID'].values
train_index = train['ID'].values

train = train.drop(['ID','target'], axis=1)
test = test.drop(['ID'], axis=1)

train, test = linear_select_interactions(train, y, 2, test, train_index, test_index)

save(train, test, train_index, test_index, y)

# valid_ids = pd.read_csv('ftrl2cv.csv')['ID'].values
# train_mask = ~np.in1d(train.ID.values, valid_ids)
# 
# 
# def encode(feature):
# 	print feature
# 	return train_1[['target',feature]].groupby(feature).target.mean().rank()
# 
# ints = train.columns[train.dtypes == np.int64]
# ints = list(ints.drop(['ID', 'target']))
# objects = list(train.columns[train.dtypes == np.object])
# columns = ints + objects
# 
# for i in range(len(columns)):
#     ci = columns[i]
#     train[ci] = train[ci].fillna("" if ci in objects else -1)
#     test[ci] = test[ci].fillna("" if ci in objects else -1)
# 
# train_1 = train[train_mask].copy()
# 
# for i in range(len(columns)):
#     ci = columns[i]
#     encode_ci = encode(ci)
#     train[ci] = train[ci].apply(lambda x: encode_ci[x] if x in encode_ci else -1)
#     test[ci] = test[ci].apply(lambda x: encode_ci[x] if x in encode_ci else -1)
#         
# train.to_csv("train.rank1.csv", index=False)
# test.to_csv("test.rank1.csv", index=False)

