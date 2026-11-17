
import time, math, random, os
import numpy as np
os.makedirs('/work', exist_ok=True)
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
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for idx,T in enumerate(temps[:4]):
    t=time.time()
    m = run(T, 600000, 1000+idx)
    np.save(f'/work/m_T{T}.npy', m)
    print(T, time.time()-t, flush=True)
