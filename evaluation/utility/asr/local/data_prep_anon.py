import sys,os

input_file = sys.argv[1]
fp = open(sys.argv[2],'w')
names = {}
for line in open(input_file):
    temp = line.strip().split('/')[-1].split('.')[0]
    if '_gen' in temp:
        temp = temp.replace('_gen','')
    fp.write('%s %s\n'%(temp,line.strip()))
fp.close()
