import csv
name='../input/train.csv'
xx=[]
for row in csv.DictReader(open(name)):
    count=0
    for x in row:
        if row[x]=='':
            count+=1
    xx.append(count)
import pickle
print sum(xx)*1.0/len(xx)
pickle.dump(xx,open('count_null.p','w'))
