
code = r'''
import numpy as np, math, random, sys, time, json
def run(N, T, seed, n_meas, n_therm, out=None):
    rng = random.Random(seed)
    L = N*N
    # neighbor table
    nb = []
    for i in range(N):
        for j in range(N):
            nb.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])
    s = [1 if rng.random()<0.5 else -1 for _ in range(L)]
    p = 1.0 - math.exp(-2.0/T)
    rand = rng.random
    M = sum(s)
    def wolff_step(s, M):
        k = int(rand()*L)
        seed_spin = s[k]
        cluster = [k]
        s[k] = -seed_spin
        stack = [k]
        while stack:
            x = stack.pop()
            for y in nb[x]:
                if s[y] == seed_spin and rand() < p:
                    s[y] = -seed_spin
                    stack.append(y)
                    cluster.append(y)
        M -= 2*seed_spin*len(cluster)
        return M
    for _ in range(n_therm):
        M = wolff_step(s, M)
    m1 = np.empty(n_meas); m2=np.empty(n_meas); m4=np.empty(n_meas)
    for t in range(n_meas):
        M = wolff_step(s, M)
        m = M / L
        m2_ = m*m
        m1[t] = abs(m); m2[t] = m2_; m4[t] = m2_*m2_
    return m1, m2, m4
'''
open('/work/wolff.py','w').write(code) if __import__('os').path.isdir('/work') else None
import os; os.makedirs('/work',exist_ok=True)
open('/work/wolff.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from wolff import run
import time
t=time.time()
m1,m2,m4 = run(16, 2.3, 1, 20000, 2000)
print("time", time.time()-t)
U = 1 - m4.mean()/(3*m2.mean()**2)
print("U", U, "<|m|>", m1.mean())
