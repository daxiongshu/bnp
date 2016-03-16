####process train and test data at the same time

#get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np


train_land=pd.read_csv('../input/train.csv').fillna(-999)
test_land=pd.read_csv('../input/test.csv').fillna(-999)
import pickle
cname=pickle.load(open('catcol.p'))
from sklearn import preprocessing
for name in cname:
    test_land[name] = map(str, test_land[name])
    train_land[name] = map(str, train_land[name])
    lbl=preprocessing.LabelEncoder()
    #train_landFeatures[name]=lbl.fit_transform(train_landFeatures[name])
    lbl.fit(pd.concat([train_land[name],test_land[name]],axis=0))
    #print lbl.classes_
    train_land[name]=lbl.transform(train_land[name])
    test_land[name]=lbl.transform(test_land[name])
    
train_land.to_csv('train_clean1.csv',index=False)

test_land.to_csv('test_clean1.csv',index=False)

print train_land.shape
print test_land.shape

