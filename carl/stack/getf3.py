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
train=pd.read_csv('../input/train.csv')#,index_col=idname)
va=pd.read_csv('../cv/mycv1.csv')#,index_col=idname)
mask=train.ID.isin(va.ID)
train=train[mask]
trainid=np.array(train.ID)
y=np.array(train.target)
train.drop('target',inplace=True,axis=1)
test=pd.read_csv('../input/test.csv')#,index_col=idname)
testid=np.array(test.ID)

X=np.vstack([y,pd.read_csv('../cv/good/xgb4.csv',index_col=idname)[labelname].values,pd.read_csv('../cv/good/ex1.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/bcv1.csv',index_col=idname).as_matrix().ravel()]).T
X=np.vstack([X.T,pd.read_csv('../srv3/cv/ridge1.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../dahei/cv/ftrl2cv.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/xgb1cv.csv',index_col=idname)['PredictedProb'].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/tree1cv.csv',index_col=idname)['PredictedProb'].values]).T

labelname='PredictedProb'
Xt=np.vstack([pd.read_csv('../cv/xgb4.csv',index_col=idname)[labelname].values,pd.read_csv('../cv/ex1.csv',index_col=idname)[labelname].values]).T
Xt=np.vstack([Xt.T,pd.read_csv('../bench/bsub1.csv',index_col=idname).as_matrix().ravel()]).T
Xt=np.vstack([Xt.T,pd.read_csv('../srv3/sub/ridge1.csv',index_col=idname)[labelname].values]).T
Xt=np.vstack([Xt.T,pd.read_csv('../dahei/sub/ftrl2sub.csv',index_col=idname)['target'].values]).T
Xt=np.vstack([Xt.T,pd.read_csv('../bench/xgb1.csv',index_col=idname)['PredictedProb'].values]).T
Xt=np.vstack([Xt.T,pd.read_csv('../bench/tree1.csv',index_col=idname)['PredictedProb'].values]).T

X=np.vstack([trainid,X.T]).T
s=pd.DataFrame(X,columns=['ID','target']+['vfea%d'%i for i in range(X.shape[1]-2)])
s=pd.merge(s,train,on='ID',how='right')
print s.shape
s.to_csv('stack_train3.csv',index=False)

Xt=np.vstack([testid,Xt.T]).T
s=pd.DataFrame(Xt,columns=['ID']+['vfea%d'%i for i in range(Xt.shape[1]-1)])
s=pd.merge(s,test,on='ID',how='right')
print s.shape

s.to_csv('stack_test3.csv',index=False)

