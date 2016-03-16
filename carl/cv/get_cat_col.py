import csv

name='../input/train.csv'
bad={}
for row in csv.DictReader(open(name)):
    for i in row:
        if row[i]=='':
            continue
        try:
            float(row[i])
        except:
            bad[i]=1
name='../input/test.csv'
for row in csv.DictReader(open(name)):
    for i in row:
        if row[i]=='':
            continue

        try:
            float(row[i])
        except:
            bad[i]=1
import pickle
print bad.keys()
pickle.dump(bad,open('catcol.p','w'))
