
import random, math, time, os
os.makedirs('/work', exist_ok=True)
src = r'''
import random, math
def make_nb(N):
    nb = []
    for i in range(N*N):
        x, y = i % N, i // N
        nb.append((((x+1)%N) + y*N, ((x-1)%N) + y*N, x + ((y+1)%N)*N, x + ((y-1)%N)*N))
    return nb

def run(N, T, nsteps, seed, nburn, nb=None, thin=1):
    rng = random.Random(seed)
    rnd = rng.random
    V = N*N
    if nb is None: nb = make_nb(N)
    s = [1 if rnd() < 0.5 else -1 for _ in range(V)]
    M = sum(s)
    padd = 1.0 - math.exp(-2.0/T)
    m2s = []; m4s = []
    total = nburn + nsteps
    for step in range(total):
        i = int(rnd()*V)
        si = s[i]
        s[i] = -si
        stack = [i]
        csize = 1
        while stack:
            j = stack.pop()
            for k in nb[j]:
                if s[k] == si and rnd() < padd:
                    s[k] = -si
                    stack.append(k)
                    csize += 1
        M -= 2*si*csize
        if step >= nburn and (step - nburn) % thin == 0:
            m = M / V
            m2 = m*m
            m2s.append(m2); m4s.append(m2*m2)
    return m2s, m4s
'''
open('/work/ising.py','w').write(src)
import sys; sys.path.insert(0,'/work')
from ising import *
t=time.time()
m2,m4 = run(16, 2.269, 20000, 1, 0)
print(time.time()-t)
import statistics
print(statistics.mean(m2), statistics.mean(m4))
