
import sys, time, numpy as np
sys.path.insert(0,'/work')
from wolff import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
res={}
for k,T in enumerate(Ts):
    m1,m2,m4 = run(16, T, 1000+k, 1_000_000, 5000)
    np.savez(f'/work/pilot_T{T}.npz', m1=m1,m2=m2,m4=m4)
    nb=500
    a2=m2.reshape(nb,-1).mean(1); a4=m4.reshape(nb,-1).mean(1)
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife
    Uj=[]
    S2=a2.sum(); S4=a4.sum()
    for b in range(nb):
        M2=(S2-a2[b])/(nb-1); M4=(S4-a4[b])/(nb-1)
        Uj.append(1-M4/(3*M2**2))
    Uj=np.array(Uj)
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    print(T, "U=%.5f err=%.5f"%(U,err), "time", round(time.time()-t0,1), flush=True)
