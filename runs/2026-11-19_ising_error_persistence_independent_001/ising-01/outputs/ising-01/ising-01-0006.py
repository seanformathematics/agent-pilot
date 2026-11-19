
import subprocess
for T,seed in [(2.3,105),(2.4,106)]:
    r=subprocess.run(['python3','/work/ising.py',str(T),str(seed),'1000','2000','20000','500'],capture_output=True,text=True,timeout=600)
    print(r.stdout.strip(), r.stderr[-500:])
import numpy as np, glob
for f in sorted(glob.glob('/work/run_*.npz')):
    d=np.load(f); S2,S4=d['S2'],d['S4']; blk=int(d['blk']); n=S2.shape[0]*blk
    a2=S2.sum(0); a4=S4.sum(0); R=len(a2)
    U=1-(a4.sum()/(R*n))/(3*(a2.sum()/(R*n))**2)
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.array([1-((tot4-a4[r])/((R-1)*n))/(3*((tot2-a2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(f, float(d['T']), round(U,6), err)
