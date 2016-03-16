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
            clf=xgb_classifier(eta=0.05,min_child_weight=20,col=0.5,subsample=0.7,depth=7,num_round=400,seed=j*77,gamma=0.1)
            y_pred+=clf.train_predict(X_train_cv,(y_train_cv),X_test_cv,y_test=(y_test_cv))
            yqq=y_pred*(1.0/(j+1))

            print j,llfun(y_test_cv,yqq)

        #y_pred/=m;
        clf=XGBClassifier(max_depth=10,colsample_bytree=0.8,learning_rate=0.02,n_estimators=500,nthread=-1)
        #clf=RandomForestClassifier(n_jobs=-1,n_estimators=100,max_depth=100)
        clf.fit(X_train_cv,(y_train_cv),eval_metric="logloss",eval_set=[(X_test_cv, y_test_cv)])
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
train=pd.read_csv('train_clean1.csv',index_col=idname).replace(-999,-1)
test=pd.read_csv('test_clean1.csv',index_col=idname).replace(-999,-1)
y=np.array(train[labelname]).astype(float)
train.drop([labelname],inplace=True,axis=1)
good=['v66', 'v34', 'v40', 'v56', 'v114', 'v3', 'v110', 'v47', 'v30', 'v99', 'v103', 'v31', 'v72', 'v10', 'v24', 'v18', 'v50', 'v55', 'v65', 'v37', 'v113', 'v77', 'v87','v62']
#train=train[good]
goodx=['v50_v129','v50_v62','v50_v15','v50_v78','v50_v88','v50_v98']
train=getgood(goodx,train)
test=getgood(goodx,test)
X=train[good+goodx].as_matrix()
Xt=test[good+goodx].as_matrix()
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
names_categorical = []
for name in train.columns.values :
    if train[name].value_counts().shape[0]<1000:# and name not in good:
        train[name] = map(str, train[name])
        test[name] = map(str, test[name])
        names_categorical.append(name)
        print name,train[name].value_counts().shape[0] 
X_sparse = vec.fit_transform(train[names_categorical].T.to_dict().values())
Xt_sparse = vec.transform(test[names_categorical].T.to_dict().values())

idx=np.array(train.index)
del train
gc.collect()
X=sparse.hstack([X,X_sparse,pd.read_csv('../online/rebuild1/train3.csv')[['v30']].as_matrix()],format='csr')#.toarray()
Xt=sparse.hstack([Xt,Xt_sparse,pd.read_csv('../online/rebuild1/test3.csv')[['v30']].as_matrix()],format='csr')
print X.shape,y.shape,Xt.shape
clf=XGBClassifier(max_depth=10,colsample_bytree=0.8,learning_rate=0.02,n_estimators=45,nthread=-1)

clf.fit(X,y)
idx=np.array(test.index)#id_test
yp=clf.predict_proba(Xt).T[1]
s=pd.DataFrame({idname:idx,'PredictedProb':yp})
s.to_csv('xgb9.csv',index=False)

