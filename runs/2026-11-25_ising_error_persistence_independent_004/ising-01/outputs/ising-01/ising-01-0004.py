
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    a,b = run(T, 500, 3000, 1000, 200+k)
    U = 1 - b.mean()/(3*a.mean()**2)
    R=len(a); idx=np.arange(R)
    Uj=np.array([1-b[idx!=r].mean()/(3*a[idx!=r].mean()**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(T, round(U,5), round(err,5), flush=True)
print("time", time.time()-t0)
