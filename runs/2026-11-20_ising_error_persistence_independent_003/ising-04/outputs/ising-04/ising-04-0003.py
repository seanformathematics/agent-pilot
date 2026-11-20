import sys, time, numpy as np
sys.path.insert(0,"/work")
from wolff import run
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
t0=time.time()
for k,T in enumerate(Ts):
    seed=1000+k
    M,c=run(16,T,seed,1000000,3000)
    np.save(f"/work/M_{T}_{seed}.npy", M)
    m2=(M/256.0)**2
    print(T, seed, "time", round(time.time()-t0,1), "U", 1-np.mean(m2**2)/(3*np.mean(m2)**2), flush=True)
