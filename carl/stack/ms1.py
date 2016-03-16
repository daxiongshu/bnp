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
    ypred=np.zeros(X_train.shape[0])
    for train_index, test_index in kf:
        count+=1
        X_train_cv, X_test_cv = X_train[train_index,:],X_train[test_index,:]
        gc.collect()
        y_train_cv, y_test_cv = y_train[train_index],y_train[test_index]
        y_pred=np.zeros(X_test_cv.shape[0])
        m=1
        for j in range(m):
            clf=xgb_classifier(eta=0.01,min_child_weight=2,col=0.7,subsample=0.8,depth=6,num_round=500,seed=j*77,gamma=0)    
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
def getgood(fea,df):
    for f in fea:
        i,j =f.split('_')
        df[f]=df[i]-df[j]
    return df
idname='ID'
labelname='target'
train=pd.read_csv('../cv/train_clean1.csv',index_col=idname).replace(-999,-1)
va=pd.read_csv('../cv/mycv1.csv',index_col=idname)
mask=train.index.isin(va.index)
train1=pd.read_csv('../online/rebuild1/train6.csv',index_col='ID')[['v113', 'v129', 'v131', 'v38', 'v24', 'v56', 'v110']]
y=np.array(train[labelname]).astype(float)
train.drop([labelname],inplace=True,axis=1)
good=['v66', 'v34', 'v40', 'v56', 'v114', 'v3', 'v110', 'v47', 'v30', 'v99', 'v103', 'v31', 'v72', 'v10', 'v24', 'v18', 'v50', 'v55', 'v65', 'v37', 'v113', 'v77', 'v87','v62','v5']
#train=train[good]
goodx=['v50_v129','v50_v62','v50_v15','v50_v78','v50_v88','v50_v98']
train=getgood(goodx,train)

X=np.hstack([train[good+goodx].as_matrix(),train1.as_matrix()])
newf=['v66-v113', 'v71-v79', 'v3-v56', 'v66-v71', 'v24-v47']
for f in newf:
    xx=f.split('-')
    train[f]=train[xx[0]].map(str)+'-'+train[xx[1]].map(str)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
names_categorical = []
cand=[]#['v40','v63','v109']#['v30']#['v5']
for name in train.columns.values :
    if train[name].value_counts().shape[0]<20 or name in cand or name in newf:# and name not in good:
        #train[name] =(train[name]*1000).astype(int)
        train[name] = map(str, train[name])
        names_categorical.append(name)
        print name,train[name].value_counts().shape[0] 
print names_categorical
X_sparse = vec.fit_transform(train[names_categorical].T.to_dict().values())
idx=np.array(train.index)
del train
gc.collect()
X=sparse.hstack([X,X_sparse],format='csr').toarray()
X,y,idx=X[np.array(mask)],y[np.array(mask)],idx[np.array(mask)]
print X.shape,y.shape,idx.shape

X=np.vstack([X.T,pd.read_csv('../cv/good/xgb4.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../cv/vabackup/mycv506874897916.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../cv/vabackup/mycv45772283918.csv',index_col=idname)[labelname].values]).T
#X=np.vstack([X.T,pd.read_csv('../cv/good/ex1.csv',index_col=idname)[labelname].values]).T
#X=np.vstack([X.T,pd.read_csv('../bench/bcv1.csv',index_col=idname).as_matrix().ravel()]).T
X=np.vstack([X.T,pd.read_csv('../srv3/cv/ridge1.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../dahei/cv/ftrl2cv.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/tree1cv.csv',index_col=idname)['PredictedProb'].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/xgb1cv.csv',index_col=idname)['PredictedProb'].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/ex4cv.csv',index_col=idname)['PredictedProb'].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/h2ocv.csv',index_col=idname)['PredictedProb'].values]).T
print X.shape,y.shape
xx=[]
yp=kfold_cv(X,y,idx,4)
s=pd.DataFrame({idname:idx,labelname:yp,'real':y})
s.to_csv('mycv1.csv',index=False)
score=str(llfun(y,yp))[2:]
import subprocess
cmd='cp mycv1.csv vabackup/mycv%s.csv'%score
subprocess.call(cmd,shell=True)
cmd='cp ms1.py vabackup/mycv%s.py'%score
subprocess.call(cmd,shell=True)
