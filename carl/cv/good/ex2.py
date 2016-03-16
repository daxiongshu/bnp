from xgb_classifier import xgb_classifier
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression,Ridge
import inspect

import os

import sys
import gc



from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer,auc,roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split,KFold,StratifiedKFold
from math import log, exp, sqrt,factorial
import numpy as np
def myauc(y,pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    return auc(fpr, tpr)
import scipy as sp
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    #ll = sum(ll)
    ll = ll * -1.0/len(act)
    return ll   
def rmspe(y,yp):
    yp[y==0]=1
    y[y==0]=1
    return np.mean(((y-yp)/y)**2)**0.5
def kfold_cv(X_train, y_train,idx,k):

    kf = StratifiedKFold(y_train,n_folds=k)
    xx=[]
    count=0
    for train_index, test_index in kf:
        count+=1
        X_train_cv, X_test_cv = X_train[train_index,:],X_train[test_index,:]
        gc.collect()
        y_train_cv, y_test_cv = y_train[train_index],y_train[test_index]
        y_pred=np.zeros(X_test_cv.shape[0])
        m=0
         
        for j in range(m):
            clf=xgb_classifier(eta=0.1,min_child_weight=20,col=0.5,subsample=0.7,depth=5,num_round=200,seed=j*77,gamma=0.1)
            y_pred+=clf.train_predict(X_train_cv,(y_train_cv),X_test_cv,y_test=(y_test_cv))
        #y_pred/=m;
        clf=ExtraTreesClassifier(n_estimators=700,max_features= 50,criterion= 'entropy',min_samples_split= 3,
                            max_depth= 60, min_samples_leaf= 4,verbose=1,n_jobs=-1)
        #clf=RandomForestClassifier(n_jobs=-1,n_estimators=100,max_depth=100)
        clf.fit(X_train_cv,(y_train_cv))
        y_pred=clf.predict_proba(X_test_cv).T[1]
        print y_pred.shape
        xx.append(llfun(y_test_cv,(y_pred)))
        ypred=y_pred
        yreal=y_test_cv
        idx=idx[test_index]
        print xx[-1]#,y_pred.shape
        break

    print xx,'average:',np.mean(xx),'std',np.std(xx)
    return ypred,yreal,idx#np.mean(xx)

def age(date):
    year=date.apply(lambda x:x.split('-')[0]).astype(int)
    month=date.apply(lambda x:x.split('-')[1]).astype(int)  
    day=date.apply(lambda x:x.split('-')[-1]).astype(int) 
    return 365*(2015-year)+31*(10-month)+31-day,year,month,day

from scipy import sparse
from sklearn import preprocessing

from scipy import sparse

from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

@mem.cache
def get_data(path):
    data = load_svmlight_file(path)
    return data[0], data[1]
idname='ID'
labelname='target'
print('Load data...')
train = pd.read_csv("../input/train.csv")
idx=np.array(train.ID)
y = train['target'].values
train = train.drop(['ID','target'],axis=1)
test = pd.read_csv("../input/test.csv")
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
for name in train.columns.values :
    train[name]=(train[name]*10).astype(int)
X = train.as_matrix()
print X.shape,y.shape
xx=[]
yp,y,idx=kfold_cv(X,y,idx,4)
s=pd.DataFrame({idname:idx,labelname:yp,'real':y})
s.to_csv('mycv1.csv',index=False)
score=str(llfun(y,yp))[2:]
import subprocess
cmd='cp mycv1.csv vabackup/mycv%s.csv'%score
subprocess.call(cmd,shell=True)
cmd='cp mycv.py vabackup/mycv%s.py'%score
subprocess.call(cmd,shell=True)
