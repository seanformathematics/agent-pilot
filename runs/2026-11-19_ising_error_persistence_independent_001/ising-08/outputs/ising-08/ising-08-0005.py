
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising2 import step
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    m2,m4 = step(T, 400, 3000, 20, 1000, 1000+k, f'/work/run_{T}.npz')
    U = 1-(m4.mean(axis=0))/(3*m2.mean(axis=0)**2)
    print(T, "U=",round(1-m4.mean()/(3*m2.mean()**2),4), "elapsed", round(time.time()-t0,1), flush=True)
