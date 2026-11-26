
import sys, time, json, numpy as np; sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
out={}
t0=time.time()
for k,T in enumerate(Ts):
    seed=5000+k
    a,b=run(16,T,seed,20000,1000000)
    a=np.array(a);b=np.array(b)
    U=1-b.mean()/(3*a.mean()**2)
    nb=200; L=len(a)//nb
    A=a[:nb*L].reshape(nb,L).mean(1); B=b[:nb*L].reshape(nb,L).mean(1)
    Sa=A.sum(); Sb=B.sum()
    Uj=np.array([1-((Sb-B[i])/(nb-1))/(3*((Sa-A[i])/(nb-1))**2) for i in range(nb)])
    err=np.sqrt((nb-1)/nb*((Uj-Uj.mean())**2).sum())
    # half-split consistency check
    h=nb//2
    Uh=[1-B[s].mean()/(3*A[s].mean()**2) for s in [slice(0,h),slice(h,nb)]]
    out[T]=dict(seed=seed,U=float(U),err=float(err),halves=[float(x) for x in Uh])
    print(T, round(U,5), round(err,5), [round(x,4) for x in Uh], round(time.time()-t0,1), flush=True)
json.dump(out,open('/work/results.json','w'),indent=1)
