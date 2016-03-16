import os
import pandas as pd
count=0
s=None
for f in os.listdir('.'):
    if 'ex' in f and 'cv.csv' in f:
        if s is None:
            s=pd.read_csv(f)
        else:
            s['PredictedProb']+=pd.read_csv(f)['PredictedProb']
        count+=1
        print f
s['PredictedProb']=s['PredictedProb']/count
s.to_csv('exave1.csv',index=False)
