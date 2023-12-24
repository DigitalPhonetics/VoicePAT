import sys,os

#generate the spk_session list according to spk
#spk_pure: spk
#spk_sid: spk_session
#spk_selec: output

# **.py spk_pure spk_sid spk_selec
spk_pure=sys.argv[1]
spk_sid = sys.argv[2]
spk_selec = sys.argv[3]

pures = []
for line in open(spk_pure):
    pures.append(line.strip())

fp = open(spk_selec,'w')
for line in open(spk_sid):
    temp = line.strip().split('-')[0]
    if temp in pures:
        fp.write('%s\n'%line.strip())
fp.close()
