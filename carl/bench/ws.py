import sys
name=sys.argv[1]
f=open('run.sh','w')
for i in range(10):
    f.write('python %s %d\n'%(name,99*i))
f.close()
