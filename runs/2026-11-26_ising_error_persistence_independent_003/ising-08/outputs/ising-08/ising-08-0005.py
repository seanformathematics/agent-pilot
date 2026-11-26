
import sys, time, json, numpy as np; sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    t=time.time()
    a,b=run(16,T,1000+k,3000,100000)
    a=np.array(a);b=np.array(b)
    U=1-b.mean()/(3*a.mean()**2)
    # block jackknife, 100 blocks
    nb=100; L=len(a)//nb
    A=a[:nb*L].reshape(nb,L).mean(1); B=b[:nb*L].reshape(nb,L).mean(1)
    Uj=np.array([1-(B.sum()-B[i])/(nb-1)/(3*((A.sum()-A[i])/(nb-1))**2) for i in range(nb)])
    err=np.sqrt((nb-1)/nb*((Uj-Uj.mean())**2).sum())
    print(T, round(U,5), round(err,5), round(time.time()-t,1), flush=True)
print('total',time.time()-t0)
