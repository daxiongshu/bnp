import pandas as pd
idname='ID'
label='target'
s1=pd.read_csv('ex3.csv',index_col=idname)
s2=pd.read_csv('../vabackup/mycv457069181263.csv',index_col=idname)
s3=pd.read_csv('ex4.csv',index_col=idname)
s1[label]=(s1[label]+s2[label]+s3[label])/3
s1.to_csv('ave2.csv')
