
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, make_nb
nb = make_nb(16)
for T in [2.2,2.3,2.4]:
    for c in range(4):
        seed = 100000 + int(round(T*100))*10 + c
        t0=time.time()
        m2,m4=run(16,T,500000,seed=seed,nburn=5000,nb=nb)
        np.savez(f'/work/chain_T{T}_c{c}.npz', m2=np.array(m2), m4=np.array(m4), seed=seed)
        print(T,c,seed,round(time.time()-t0,1),flush=True)
