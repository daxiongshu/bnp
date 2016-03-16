import pandas as pd
idname='ID'
label='target'
s1=pd.read_csv('ex3.csv',index_col=idname)
s2=pd.read_csv('../vabackup/mycv457069181263.csv',index_col=idname)
s1[label]=(s1[label]+s2[label])/2
s1.to_csv('ave1.csv')
