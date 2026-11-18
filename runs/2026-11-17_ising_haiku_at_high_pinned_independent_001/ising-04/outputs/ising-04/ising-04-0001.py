
import time, importlib
try:
    import numba; print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)
import numpy as np, random
N=16
def wolff_run(T, nsweeps, seed):
    rng = random.Random(seed)
    beta = 1.0/T
    p = 1-np.exp(-2*beta)
    s = [1]*(N*N)
    # neighbor table
    nb = []
    for i in range(N):
        for j in range(N):
            k=i*N+j
            nb.append((((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N))
    ms=[]
    for _ in range(nsweeps):
        seed_site = rng.randrange(N*N)
        sp = s[seed_site]
        cl=[seed_site]; s[seed_site]=-sp
        stack=[seed_site]
        while stack:
            k=stack.pop()
            for n in nb[k]:
                if s[n]==sp and rng.random()<p:
                    s[n]=-sp; stack.append(n); cl.append(n)
        ms.append(sum(s))
    return ms
t=time.time(); ms=wolff_run(2.269,20000,1); print("time per 20000 clusters", time.time()-t)
