
import random, math, time
import numpy as np
def wolff_run(N, T, nclus, seed, nbin=1000, burn=2000):
    rng = random.Random(seed)
    V = N*N
    nb = [0]*(4*V)
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[4*i+0] = ((x+1)%N)*N + y
            nb[4*i+1] = ((x-1)%N)*N + y
            nb[4*i+2] = x*N + (y+1)%N
            nb[4*i+3] = x*N + (y-1)%N
    s = [1]*V
    M = V
    padd = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    out = []
    acc2=acc4=accm=0.0; cnt=0
    for step in range(burn+nclus):
        i0 = int(rnd()*V)
        sd = s[i0]
        s[i0] = -sd
        stack = [i0]
        size = 1
        while stack:
            j = stack.pop()
            for k in (nb[4*j], nb[4*j+1], nb[4*j+2], nb[4*j+3]):
                if s[k] == sd and rnd() < padd:
                    s[k] = -sd
                    stack.append(k)
                    size += 1
        M += -2*sd*size
        if step >= burn:
            m = M/V
            m2 = m*m
            acc2 += m2; acc4 += m2*m2; accm += m; cnt += 1
            if cnt == nbin:
                out.append((acc2, acc4, accm, cnt))
                acc2=acc4=accm=0.0; cnt=0
    return np.array(out)
def U_of(b):
    m2 = b[:,0].sum()/b[:,3].sum(); m4 = b[:,1].sum()/b[:,3].sum()
    return 1 - m4/(3*m2*m2)
def jack(b):
    n = len(b); full = U_of(b)
    loo = np.array([U_of(np.delete(b,i,axis=0)) for i in range(n)])
    return full, math.sqrt((n-1)/n*np.sum((loo-loo.mean())**2))
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    b = wolff_run(16, T, 400000, seed=910000+int(T*1000))
    u,e = jack(b)
    print(T, round(u,5), round(e,5), flush=True)
