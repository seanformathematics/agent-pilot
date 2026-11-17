
import time, math, random, os
import numpy as np
N=16
NN=N*N
nb = []
for i in range(N):
    for j in range(N):
        nb.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])
def run(T, nclus, seed, burn=20000):
    rng = random.Random(seed)
    p = 1-math.exp(-2.0/T)
    s = [1]*NN
    M = NN
    rnd = rng.random; rr = rng.randrange
    out = np.empty(nclus, dtype=np.int32)
    for it in range(nclus+burn):
        k0 = rr(NN)
        st = s[k0]; s[k0] = -st
        stack=[k0]; M -= 2*st
        while stack:
            k=stack.pop()
            for n in nb[k]:
                if s[n]==st and rnd()<p:
                    s[n]=-st; stack.append(n); M -= 2*st
        if it>=burn:
            out[it-burn]=M
    return out
for idx,T in zip([5,6,7,8],[1.9,2.0,2.1,2.2]):
    t=time.time()
    m = run(T, 2000000, 2000+idx)
    np.save(f'/work/m_T{T}.npy', m)
    print(T, time.time()-t, flush=True)
