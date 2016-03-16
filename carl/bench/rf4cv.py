import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import log_loss, make_scorer
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV
import random
random.seed(16)

df_train = pd.read_csv('../input/tr.csv') #.fillna(-999)
df_test = pd.read_csv('../input/va.csv') #.fillna(-999)
num_train = df_train.shape[0]
target_val = 'target'
id_test = df_test['ID']
y_train = df_train['target']
d_col_drops=['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
test = df_test.drop(d_col_drops,axis=1)
d_col_drops.append('target')
train = df_train.drop(d_col_drops,axis=1)

def flog_loss(ground_truth, predictions):
    flog_loss_ = log_loss(ground_truth, predictions) #, eps=1e-15, normalize=True, sample_weight=None)
    return flog_loss_

LL  = make_scorer(flog_loss, greater_is_better=False)

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

rfr = RandomForestRegressor(n_estimators = 20, n_jobs = -1, random_state = 2016, verbose = 0)
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [15]}
model = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 0, scoring=LL)
model.fit(train, y_train.values)

print("Best CV score: ", model.best_score_)
y_pred = model.predict(test)

min_y_pred = min(y_pred)
max_y_pred = max(y_pred)
min_y_train = min(y_train.values)
max_y_train = max(y_train.values)
print(min_y_pred, max_y_pred, min_y_train, max_y_train)
for i in range(len(y_pred)):
    if y_pred[i]<0.0:
        y_pred[i] = 0.0
    if y_pred[i]>1.0:
        y_pred[i] = 1.0

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred}).to_csv('rf4cv1.csv',index=False)
print("--- Training & Testing RFR: %s minutes ---" % round(((time.time() - start_time)/60),2))

extc = ExtraTreesClassifier(n_estimators=850,max_features= 60,criterion= 'entropy',min_samples_split= 4, max_depth= 40, min_samples_leaf= 2, n_jobs = -1)      
extc.fit(train, y_train.values) 
y_pred2 = extc.predict_proba(test)

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred2[:,1]}).to_csv('rf4cv2.csv',index=False)

y_pred3 = (y_pred + y_pred2[:,1])/2
pd.DataFrame({"ID": id_test, "PredictedProb": y_pred3}).to_csv('rf4cv3.csv',index=False)
