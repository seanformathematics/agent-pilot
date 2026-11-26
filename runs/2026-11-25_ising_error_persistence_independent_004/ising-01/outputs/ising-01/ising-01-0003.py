
import sys, time, numpy as np, json
sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    a,b = run(T, 1000, 4000, 1000, 100+k)
    U_rep = 1 - b/(3*a*a)
    U = 1 - b.mean()/(3*a.mean()**2)
    # jackknife over replicas
    R=len(a); idx=np.arange(R)
    Uj=[]
    for r in range(R):
        mask=idx!=r
        aa=a[mask].mean(); bb=b[mask].mean()
        Uj.append(1-bb/(3*aa*aa))
    Uj=np.array(Uj)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    res[T]=(U,err)
    print(T, round(U,5), round(err,5), round(a.mean(),5), flush=True)
print("time", time.time()-t0)
