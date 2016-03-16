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

idname='ID'
labelname='PredictedProb'

sub1=sys.argv[1]
sub2=sys.argv[2]

tmp1=pd.read_csv(sub1,index_col=idname)
tmp2=pd.read_csv(sub2,index_col=idname)

y1=np.array(tmp1[labelname])#.argsort().argsort()*1.0/tmp1.shape[0]
y2=np.array(tmp2[labelname])

y=np.array(pd.read_csv('../cv/good/xgb11.csv')['real'])
bests=0
besty=None
for i in range(11):
    #yp=y1.argsort().argsort()*i+y2.argsort().argsort()*(10-i)
    #yp=yp*1.0/y1.shape[0]
    yp=y1*i+y2*(10-i)
    yp/=10
    tmp=llfun(y,yp)
    if tmp>bests:
        bests=tmp
        besty=yp.copy()
    print i,tmp
if len(sys.argv)>3:
    b=pd.read_csv(sub1)
    b[name]=besty
    b.to_csv('com2.csv',index=False)
#b.to_csv(sub1.split('.')[0]+'_'+sub2.split('.')[0]+'.csv')
