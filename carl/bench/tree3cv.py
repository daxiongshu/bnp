# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import random


from sklearn import metrics

rnd=57
maxCategories=10

train=pd.read_csv('../input/tr.csv')
test=pd.read_csv('../input/va.csv')
random.seed(rnd)
train.index=train.ID
test.index=test.ID
del train['ID'], test['ID']
target=train.target
del train['target']


train = train.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
test = test.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)


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
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999
            

print('Training...')
    
#run cross validation
from sklearn import cross_validation
X, Y, Xtarget, Ytarget=cross_validation.train_test_split(train, target, test_size=0.25,random_state=42)

print("-"*53)

from sklearn import ensemble

clfs=[
    
    ensemble.RandomForestClassifier(bootstrap=False, class_weight='auto', criterion='entropy',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
            oob_score=False, random_state=rnd, verbose=0,
            warm_start=False),
    ensemble.ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
           max_depth=None, max_features='sqrt', max_leaf_nodes=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=1e-5, n_estimators=500, n_jobs=-1,
           oob_score=False, random_state=rnd, verbose=0, warm_start=False),
           
    ensemble.GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=2, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=3,
              min_weight_fraction_leaf=0.0, n_estimators=50,
              presort='auto', random_state=rnd, subsample=1.0, verbose=0,
              warm_start=False)
]

indice=0
preds=[]
predstest=[]

#run models
for model in clfs:
    
    model.fit(X, Xtarget)

    preds.append(model.predict_proba(Y)[:,1])
    print('model ',indice,': loss=',metrics.log_loss(Ytarget,preds[indice]))

    noms=pd.DataFrame(test.columns[abs(model.feature_importances_)>1e-10][:30])
    noms.columns=['noms']
    coefs=pd.DataFrame(model.feature_importances_[abs(model.feature_importances_)>1e-10][:30])
    coefs.columns=['coefs']
    df=pd.concat([noms, coefs], axis=1).sort_values(by=['coefs'])

    #plt.figure(indice)
    #df.plot(kind='barh', x='noms', y='coefs', legend=True, figsize=(6, 10))
    #plt.savefig('clf'+str(indice)+'_ft_importances.jpg')

    predstest.append(model.predict_proba(test)[:,1])
    indice+=1

#find best weights
step=0.1 * (1./len(preds))
print("step:", step)
poidsref=np.zeros(len(preds))
poids=np.zeros(len(preds))
poidsreftemp=np.zeros(len(preds))
poidsref=poidsref+1./len(preds)

bestpoids=poidsref.copy()
blend_cv=np.zeros(len(preds[0]))

for k in range(0,len(preds),1):
    blend_cv=blend_cv+bestpoids[k]*preds[k]
bestscore=metrics.log_loss(Ytarget.values,blend_cv)

getting_better_score=True
while getting_better_score:
    getting_better_score=False
    for i in range(0,len(preds),1):
        poids=poidsref
        if poids[i]-step>-step:
            #decrease weight in position i
            poids[i]-=step
            for j in range(0,len(preds),1):
                if j!=i:
                    if poids[j]+step<=1:
                        #try an increase in position j
                        poids[j]+=step
                        #score new weights
                        blend_cv=np.zeros(len(preds[0]))
                        for k in range(0,len(preds),1):
                            blend_cv=blend_cv+poids[k]*preds[k]
                        actualscore=metrics.log_loss(Ytarget.values,blend_cv)
                        #if better, keep it
                        if actualscore<bestscore:
                            bestscore=actualscore
                            bestpoids=poids.copy()
                            getting_better_score=True
                        poids[j]-=step
            poids[i]+=step
    poidsref=bestpoids.copy()

print("weights: ", bestpoids)
print("optimal blend loss: ", bestscore)


blend_to_submit=np.zeros(len(predstest[0]))

for i in range(0,len(preds),1):
    blend_to_submit=blend_to_submit+bestpoids[i]*predstest[i]

#submit
submission=pd.read_csv('ex5cv.csv')
submission.PredictedProb=blend_to_submit
submission.to_csv('tree3cv.csv', index=False)
