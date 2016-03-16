from sklearn.metrics import mean_squared_error
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split,KFold
import scipy as sp
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    #ll = sum(ll)
    ll = ll * -1.0/len(act)
    return ll

def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
labelname='PredictedProb'
idname='ID'

sub1=sys.argv[1]

tmp1=pd.read_csv(sub1,index_col=idname).fillna(0.76)

tmp2=pd.read_csv('../cv/good/xgb4.csv',index_col=idname)
y=np.array(tmp2['real'])[:tmp1.shape[0]]
yp=np.array(tmp1[labelname])#.argsort().argsort()*1.0/tmp1.shape[0]
#print 'auc',myauc(y,yp)
print 'logloss',llfun(y,yp)

