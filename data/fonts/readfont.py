import os

indir = './CN'
files = os.listdir(indir)

with open('fontlist.txt','w') as f:
    for filename in files:
        f.write('CN/'+filename+'\n')
