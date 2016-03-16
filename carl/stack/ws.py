fea=['v3', 'v24', 'v30', 'v31', 'v38', 'v47', 'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125', 'v129']
f=open('run.sh','w')
for ff in fea:
    f.write('python ms.py %s\n'%ff)
f.close()
