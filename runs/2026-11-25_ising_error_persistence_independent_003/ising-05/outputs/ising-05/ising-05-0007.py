
import numpy as np, time, sys
sys.path.insert(0,'/work')
from sw import *
seeds={2.2:1004,2.3:1005,2.4:1006}
for T in [2.2,2.3,2.4]:
    t0=time.time()
    M2,M4,M1=run(T,400,200,2000,seeds[T])
    u,e=binder_jk(M2,M4)
    np.savez(f'/work/prod_T{T}.npz',M2=M2,M4=M4,M1=M1)
    print(T, u, e, "time", round(time.time()-t0,1), flush=True)
