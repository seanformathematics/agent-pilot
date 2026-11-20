
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    r=np.array(run(16,T,200000,1000+k,2000))
    np.save(f'/work/pilot_{T}.npy',r)
    m2=np.mean(r**2); m4=np.mean(r**4)
    U=1-m4/(3*m2**2)
    # block jackknife
    nb=200
    bl=np.array_split(r,nb)
    vals=[]
    for i in range(nb):
        rest=np.concatenate([bl[j] for j in range(nb) if j!=i])
        vals.append(1-np.mean(rest**4)/(3*np.mean(rest**2)**2))
    vals=np.array(vals)
    err=np.sqrt((nb-1)/nb*np.sum((vals-vals.mean())**2))
    print(T, round(U,5), round(err,5), 'mean|m|',round(np.mean(np.abs(r)),4), 'elapsed',round(time.time()-t0,1))
