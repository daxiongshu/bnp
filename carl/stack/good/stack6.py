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
train=pd.read_csv('../cv/train_clean1.csv',index_col=idname)
va=pd.read_csv('../cv/mycv1.csv',index_col=idname)
mask=train.index.isin(va.index)
train=train[mask]
#trainx=pd.read_csv('../input/train.csv',index_col=idname).fillna('2015-12-31')
#print train.shape, trainx.shape
#x,y,z,_=age(trainx['Original_Quote_Date'])
#train['Original_Quote_Date']=x
#train['year']=y
#train['month']=z

#corr < 1e-2
bad=['v2', 'v4', 'v6', 'v7', 'v17', 'v26', 'v27', 'v43', 'v44', 'v60', 'v64', 'v76', 'v82', 'v87', 'v88', 'v93', 'v99', 'v100', 'v101', 'v102', 'v105', 'v106', 'v108', 'v109', 'v116', 'v127', 'v128', 'v131']
y=np.array(train[labelname]).astype(float)
train.drop([labelname]+bad,inplace=True,axis=1)

X=train.as_matrix()

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
names_categorical = []
for name in train.columns.values :
    if train[name].value_counts().shape[0]<1000:
        train[name] = map(str, train[name])
        names_categorical.append(name)
        print name,train[name].value_counts().shape[0] 
X_sparse = vec.fit_transform(train[names_categorical].T.to_dict().values())
idx=np.array(train.index)
del train
gc.collect()

print X.shape,y.shape
X=np.vstack([X.T,pd.read_csv('../cv/good/xgb4.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../cv/good/ex1.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/bcv1.csv',index_col=idname).as_matrix().ravel()]).T
X=np.vstack([X.T,pd.read_csv('../srv3/cv/ridge1.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../dahei/cv/ftrl2cv.csv',index_col=idname)[labelname].values]).T
X=np.vstack([X.T,pd.read_csv('../bench/tree1cv.csv',index_col=idname)['PredictedProb'].values]).T

print X.shape,y.shape
X=sparse.hstack([X,X_sparse],format='csr')#.toarray()
print X.shape,y.shape
xx=[]
yp=kfold_cv(X,y,idx,4)
s=pd.DataFrame({idname:idx,labelname:yp,'real':y})
s.to_csv('mycv1.csv',index=False)
score=str(llfun(y,yp))[2:]
import subprocess
cmd='cp mycv1.csv vabackup/mycv%s.csv'%score
subprocess.call(cmd,shell=True)
cmd='cp ms.py vabackup/mycv%s.py'%score
subprocess.call(cmd,shell=True)
