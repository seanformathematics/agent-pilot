
import random, math, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import random, math, sys, time
import numpy as np

def make_nb(N):
    nb = []
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb.append((((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N))
    return nb

def wolff_run(N, T, nsteps, seed, nburn=2000):
    rng = random.Random(seed)
    nb = make_nb(N)
    V = N*N
    p = 1.0 - math.exp(-2.0/T)
    s = [1 if rng.random()<0.5 else -1 for _ in range(V)]
    M = sum(s)
    rnd = rng.random
    def step():
        nonlocal M
        i0 = int(rnd()*V)
        si = s[i0]
        s[i0] = -si
        stack = [i0]
        cnt = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j] == si and rnd() < p:
                    s[j] = -si
                    stack.append(j)
                    cnt += 1
        M -= 2*si*cnt
        return
    for _ in range(nburn):
        step()
    m2 = np.empty(nsteps); m4 = np.empty(nsteps)
    for k in range(nsteps):
        step()
        m = M / V
        m2_ = m*m
        m2[k] = m2_
        m4[k] = m2_*m2_
    return m2, m4
'''
open('/work/wolff.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from wolff import wolff_run
t=time.time()
m2,m4 = wolff_run(16, 2.269, 20000, 1, nburn=500)
dt=time.time()-t
print("time per step us", dt/20500*1e6)
print("U", 1-m4.mean()/(3*m2.mean()**2))
