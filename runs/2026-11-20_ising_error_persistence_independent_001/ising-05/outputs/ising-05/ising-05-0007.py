import sys, time, numpy as np
sys.path.insert(0,'/work')
from wolff import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for k,T in enumerate(Ts):
    t0=time.time(); m2,m4=run(16,T,2000000,100+k,5000)
    np.savez(f'/work/d1_{T}.npz',m2=m2,m4=m4)
    print(T, round(time.time()-t0,1), 1-m4.mean()/(3*m2.mean()**2), flush=True)
