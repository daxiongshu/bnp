from xgb_classifier import xgb_classifier
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn import cross_validation
import inspect
import random
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
    ypred=np.zeros(X_train.shape[0])
    for train_index, test_index in kf:
        count+=1
        X_train_cv, X_test_cv = X_train[train_index,:],X_train[test_index,:]
        gc.collect()
        y_train_cv, y_test_cv = y_train[train_index],y_train[test_index]
        y_pred=np.zeros(X_test_cv.shape[0])
        m=1
         
        for j in range(m):
            clf=xgb_classifier(eta=0.01,min_child_weight=10,col=0.7,subsample=0.68,depth=5,num_round=500,seed=j*77,gamma=0)

            y_pred+=clf.train_predict(X_train_cv,(y_train_cv),X_test_cv,y_test=(y_test_cv))
            yqq=y_pred/(1+j)
            print j,llfun(y_test_cv,yqq)
        y_pred/=m;
        #clf=RandomForestClassifier(n_jobs=-1,n_estimators=100,max_depth=100)
        #clf.fit(X_train_cv,(y_train_cv))
        #y_pred=clf.predict_proba(X_test_cv).T[1]
        print y_pred.shape
        xx.append(llfun(y_test_cv,(y_pred)))
        ypred[test_index]=y_pred
        print xx[-1]#,y_pred.shape

    print xx,'average:',np.mean(xx),'std',np.std(xx)
    return ypred
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


# X, y = load_svmlight_file("meta/meta-train-sparse.svm")
# Xt, _ = load_svmlight_file("meta/meta-test-sparse.svm")

y = pd.read_csv('ftrl2cv.csv')['real'].values
test = pd.read_csv('meta/meta-test.csv')
train = pd.read_csv('meta/meta-train.csv')

idx = test.ID.astype(int).values

del test['ID']
del train['ID']

X = train.values
Xt = test.values

X = np.vstack([X.T,pd.read_csv('eval2/et1_20160316185944[0.468351495946].csv',index_col=idname)['PredictedProb'].values]).T
Xt = np.vstack([Xt.T,pd.read_csv('test2/et1_20160316185944[0.468351495946].csv',index_col=idname)['PredictedProb'].values]).T

X = np.vstack([X.T,pd.read_csv('eval2/xgb10b_20160316191957[0.455228169043].csv',index_col=idname)['PredictedProb'].values]).T
Xt = np.vstack([Xt.T,pd.read_csv('test2/xgb10b_20160316191957[0.455228169043].csv',index_col=idname)['PredictedProb'].values]).T

print X.shape, Xt.shape


bad=[8,114]
xx=[i for i in range(X.shape[1]) if i not in bad]
X, Xt = X[:,xx], Xt[:,xx]

assert(X.shape[1]==Xt.shape[1])



# kf = cross_validation.StratifiedKFold(y, 4, shuffle=True, random_state=11)
# 
# cvscores = []
# for itr, icv in kf:
# 	X_train, X_valid    = X[itr], 	X[icv]
# 	y_train, y_valid    = y[itr],   y[icv]
# 
# 	clf=xgb_classifier(eta=0.01,min_child_weight=2,col=0.7,subsample=0.68,depth=5,num_round=500,seed=0,gamma=0)
# 	y_pred=clf.train_predict(X_train, y_train, X_valid, y_valid)
# 	
# 	cvscores.append(llfun(y_valid,y_pred))
# 	
# print cvscores, np.mean(cvscores)
# sys.exit(0)

clf=xgb_classifier(eta=0.01,min_child_weight=2,col=0.7,subsample=0.68,depth=5,num_round=600,seed=0,gamma=0)
yp=clf.train_predict(X, y, Xt)
s=pd.DataFrame({idname:idx,'PredictedProb':yp})
s.to_csv('stack32.csv',index=False)

