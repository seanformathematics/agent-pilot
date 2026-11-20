
import sys, json, glob; sys.path.insert(0,'/work')
import numpy as np, sim
for f in sorted(glob.glob('/work/raw_T*.json')):
    d=json.load(open(f)); S2=np.array(d['S2']); S4=np.array(d['S4'])
    nb=len(S2); n=2000
    Us=[]
    for c in np.array_split(np.arange(nb),4):
        Us.append(sim.Uof(S2[c].sum(),S4[c].sum(),len(c)*n))
    full,err=sim.analyse(S2,S4,n)
    print(d['T'], 'U=%.6f err=%.2e'%(full,err), 'quarters:', ['%.5f'%u for u in Us], 'nbins',nb)
