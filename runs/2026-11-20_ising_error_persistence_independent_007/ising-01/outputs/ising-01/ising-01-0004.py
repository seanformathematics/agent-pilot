import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run
for T in [2.2,2.3,2.4]:
    t=time.time()
    s,M2,M4,Ma = run(T,1000,16,2000,2000,10,1000+int(round(T*100)))
    np.savez('/work/ts_%.2f.npz'%T, M2=M2.astype(np.float32), M4=M4.astype(np.float32), Ma=Ma.astype(np.float32))
    U = 1-M4.mean()/(3*M2.mean()**2)
    print("T=%.2f U=%.5f  time=%.1f"%(T,U,time.time()-t), flush=True)
