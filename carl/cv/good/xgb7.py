#https://www.kaggle.com/anilkumarkuppa/bnp-paribas-cardif-claims-management/extratreesclassifier-score-0-45911
import sys
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import scipy as sp
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    #ll = sum(ll)
    ll = ll * -1.0/len(act)
    return ll

print('Load data...')
train = pd.read_csv("../input/tr.csv")
target = train['target'].values
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv("../input/va.csv")
id_test = test['ID'].values
test = test.drop(['ID'],axis=1)

print('Clearing...')
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
            train.loc[train_series.isnull(), train_name] = -9999 #train_series.mean()
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -9999 #train_series.mean()  #TODO

X_train = train
X_test = test

extc = XGBClassifier(max_depth=10,colsample_bytree=0.8,learning_rate=0.02,n_estimators=500,nthread=-1)#max_features= 50,criterion= 'entropy',min_samples_split= 4,
                            #max_depth= 50, min_samples_leaf= 4)      
y_test=pd.read_csv('good/xgb4.csv')['real'].values

extc.fit(X_train,target,eval_metric="logloss",eval_set=[(X_test, y_test)]) 

print('Predict...')
y_pred = extc.predict_proba(X_test)
#print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('mycv1.csv',index=False)
y=pd.read_csv('good/xgb4.csv')['real'].values
yp=y_pred[:,1]
score=str(llfun(y,yp))[2:]
print sys.argv[0],score
import subprocess
cmd='cp mycv1.csv vabackup/mycv%s.csv'%score
subprocess.call(cmd,shell=True)
cmd='cp mycv.py vabackup/mycv%s.py'%score
subprocess.call(cmd,shell=True)

