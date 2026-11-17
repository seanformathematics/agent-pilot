
import time, importlib
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import numpy as np
N=16
def wolff_run(T, nsteps, seed):
    rng=np.random.default_rng(seed)
    s=np.ones(N*N,dtype=np.int8)
    p=1-np.exp(-2.0/T)
    nb=np.zeros((N*N,4),dtype=np.int64)
    for i in range(N):
        for j in range(N):
            k=i*N+j
            nb[k]=[((i+1)%N)*N+j,((i-1)%N)*N+j,i*N+(j+1)%N,i*N+(j-1)%N]
    nb=nb.tolist(); sl=s.tolist()
    rnd=rng.random(10_000_000).tolist(); ri=0
    ms=[]
    for step in range(nsteps):
        st=int(rng.integers(N*N))
        c=sl[st]; sl[st]=-c; stack=[st]; 
        while stack:
            k=stack.pop()
            for q in nb[k]:
                if sl[q]==c:
                    if rnd[ri]<p:
                        sl[q]=-c; stack.append(q)
                    ri+=1
                    if ri>=10_000_000: ri=0
        ms.append(sum(sl))
    return ms
t=time.time(); ms=wolff_run(2.269,20000,1); print(time.time()-t)
