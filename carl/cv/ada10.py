from xgb_classifier import xgb_classifier
from xgboost import XGBClassifier

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
            clf=xgb_classifier(eta=0.05,min_child_weight=20,col=0.3,subsample=0.7,depth=7,num_round=400,seed=j*77,gamma=0.1)
            y_pred+=clf.train_predict(X_train_cv,(y_train_cv),X_test_cv,y_test=(y_test_cv))
            yqq=y_pred*(1.0/(j+1))

            print j,llfun(y_test_cv,yqq)
        #clf=ExtraTreesClassifier(n_estimators=850,max_features= 60,criterion= 'entropy',min_samples_split= 4,
        #                    verbose=1,max_depth= 40, min_samples_leaf= 2, n_jobs = -1)
        clf=AdaBoostClassifier(
            n_estimators = 20,
            learning_rate = 0.75,
            base_estimator = ExtraTreesClassifier(n_estimators=20,max_features= 60,criterion= 'entropy',min_samples_split= 4,
                max_depth= 40, min_samples_leaf= 2,
                verbose = 1,
                n_jobs = -1))
        #y_pred/=m;
        #clf=XGBClassifier(max_depth=12,colsample_bytree=0.3,learning_rate=0.01,n_estimators=750,nthread=-1)
        #clf=RandomForestClassifier(n_jobs=-1,n_estimators=100,max_depth=100)
        clf.fit(X_train_cv,(y_train_cv))#,eval_metric="logloss",eval_set=[(X_test_cv, y_test_cv)])
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
def getgood(fea,df):
    for f in fea:
        i,j =f.split('_')
        df[f]=df[i]-df[j]
    return df
idname='ID'
labelname='target'
train=pd.read_csv('../tian/clean/train_clean3.csv',index_col=idname).replace(-999,-1)
y=np.array(train[labelname]).astype(float)
train.drop([labelname],inplace=True,axis=1)
X=train.as_matrix()
idx=np.array(train.index)
train=pd.read_csv('../tian/clean/test_clean3.csv',index_col=idname).replace(-999,-1)
train.drop([labelname],inplace=True,axis=1)
Xt=train.as_matrix()
idx=np.array(train.index)
print X.shape,y.shape,Xt.shape
clf=AdaBoostClassifier(
            n_estimators = 20,
            learning_rate = 0.75,
            base_estimator = ExtraTreesClassifier(n_estimators=20,max_features= 200,criterion= 'entropy',min_samples_split= 4,
                max_depth= 40, min_samples_leaf= 2,
                verbose = 1,
                n_jobs = -1))
clf.fit(X,y)
yp=clf.predict_proba(Xt).T[1]
s=pd.DataFrame({idname:idx,'PredictedProb':yp})
s.to_csv('ada10.csv',index=False)
