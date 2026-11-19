
import subprocess, numpy as np
r=subprocess.run(['python3','/work/ising.py','2.3','205','500','6000','10000','500'],capture_output=True,text=True,timeout=600)
print(r.stdout.strip(), r.stderr[-300:])
def U_of(f):
    d=np.load(f); S2,S4=d['S2'],d['S4']; blk=int(d['blk']); n=S2.shape[0]*blk
    a2=S2.sum(0); a4=S4.sum(0); R=len(a2)
    U=1-(a4.sum()/(R*n))/(3*(a2.sum()/(R*n))**2)
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.array([1-((tot4-a4[k])/((R-1)*n))/(3*((tot2-a2[k])/((R-1)*n))**2) for k in range(R)])
    return U, np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
print("main T=2.3:",U_of('/work/run_T2.300_s105.npz'))
print("check T=2.3:",U_of('/work/run_T2.300_s205.npz'))
