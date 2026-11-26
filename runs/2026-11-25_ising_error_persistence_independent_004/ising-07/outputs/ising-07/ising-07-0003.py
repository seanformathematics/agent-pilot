
import sys, time, numpy as np, json
sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m):
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2*m2)
res={}
t0=time.time()
for k,T in enumerate(Ts):
    m=np.array(run(16,T,3000,500000,1000+k))
    a=np.abs(m)
    U=U_of(a) if False else U_of(m)
    nb=500
    bs=len(m)//nb
    Ub=[U_of(m[i*bs:(i+1)*bs]) for i in range(nb)]
    err=np.std(Ub,ddof=1)/np.sqrt(nb)
    res[T]=(U,err)
    print(T, round(U,5), round(err,5), "m2",np.mean(m**2), round(time.time()-t0,1), flush=True)
