
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m):
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2*m2)
t0=time.time()
allU={}
for k,T in enumerate(Ts):
    Us=[]
    for r in range(2):
        m=np.array(run(16,T,3000,500000,2000+10*k+r))
        Us.append(U_of(m))
    allU[T]=Us
    print(T, [round(u,5) for u in Us], round(time.time()-t0,1), flush=True)
