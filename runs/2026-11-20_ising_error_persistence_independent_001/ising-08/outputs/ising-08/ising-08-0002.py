import random, math, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import random, math, time, sys, json
import numpy as np
def run(N, T, seed, seconds, nburn=2000, nbin=200):
    rng = random.Random(seed)
    V = N*N
    nb = []
    for i in range(V):
        x, y = i % N, i // N
        nb.append((((x+1)%N) + y*N, ((x-1)%N) + y*N, x + ((y+1)%N)*N, x + ((y-1)%N)*N))
    s = [1 if rng.random()<0.5 else -1 for _ in range(V)]
    p = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    M = sum(s)
    def wolff_step(s, M):
        i0 = int(rnd()*V)
        si = s[i0]
        s[i0] = -si
        stack = [i0]
        cl = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j] == si and rnd() < p:
                    s[j] = -si
                    stack.append(j)
                    cl += 1
        return cl*(-2*si) if False else (M - 2*si*cl)
    # burn-in
    for _ in range(nburn):
        M = wolff_step(s, M)
    m2s = []; m4s = []
    t0 = time.time()
    n = 0
    while time.time()-t0 < seconds:
        M = wolff_step(s, M)
        m = M / V
        m2 = m*m
        m2s.append(m2); m4s.append(m2*m2)
        n += 1
    m2s = np.array(m2s); m4s = np.array(m4s)
    return n, m2s, m4s
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
import numpy as np
t=time.time()
n,m2,m4 = run(16, 2.269, 1, 20)
print("updates", n, "rate", n/20)
U = 1 - m4.mean()/(3*m2.mean()**2)
print("U", U, "<m2>",m2.mean())
# naive err
print("naive err", np.std(1-m4/(3*m2.mean()**2))/np.sqrt(n))
