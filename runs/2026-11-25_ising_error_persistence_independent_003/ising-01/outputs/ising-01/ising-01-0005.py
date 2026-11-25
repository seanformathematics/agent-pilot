import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, binder_jk
for T,seed in [(1.9,201),(2.0,202),(2.1,203)]:
    t=time.time()
    S2,S4,cnt=run(T,500,20000,500,seed,block=200)
    np.savez(f'/work/T{T}_s{seed}.npz',S2=S2,S4=S4,cnt=cnt)
    print(T,"time",round(time.time()-t,1), binder_jk(S2,S4,cnt), flush=True)
