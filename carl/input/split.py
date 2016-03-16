import pandas as pd
train=pd.read_csv('train.csv')
vaid=pd.read_csv('../cv/mycv1.csv')
mask=train.ID.isin(vaid.ID)
train[mask].drop('target',axis=1).to_csv('va.csv',index=False)
train[~mask].to_csv('tr.csv',index=False)

