
code = r'''
import random, math, sys, time, json, os
import numpy as np

N = 16
NS = N*N
nb = []
for x in range(N):
    for y in range(N):
        i = x*N+y
        nb.append((((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N))

def run(T, seed, n_clusters, block, state=None, time_budget=1e9):
    """Wolff updates; accumulates per-block sums of m^2, m^4 (per-cluster measurements)."""
    rng = random.Random(seed)
    rnd = rng.random
    padd = 1.0 - math.exp(-2.0/T)
    if state is None:
        s = [1 if rnd()<0.5 else -1 for _ in range(NS)]
        M = sum(s)
        thermal = 3000
    else:
        s, M = state
        thermal = 0
    t0 = time.time()
    # thermalization
    for _ in range(thermal):
        i = int(rnd()*NS)
        sg = s[i]
        s[i] = -sg
        stack = [i]; size = 1
        while stack:
            j = stack.pop()
            for k in nb[j]:
                if s[k] == sg and rnd() < padd:
                    s[k] = -sg; stack.append(k); size += 1
        M -= 2*sg*size
    blocks_m2 = []; blocks_m4 = []
    done = 0
    while done < n_clusters:
        b2 = 0.0; b4 = 0.0
        for _ in range(block):
            i = int(rnd()*NS)
            sg = s[i]
            s[i] = -sg
            stack = [i]; size = 1
            while stack:
                j = stack.pop()
                for k in nb[j]:
                    if s[k] == sg and rnd() < padd:
                        s[k] = -sg; stack.append(k); size += 1
            M -= 2*sg*size
            m = M/NS
            m2 = m*m
            b2 += m2; b4 += m2*m2
        blocks_m2.append(b2/block); blocks_m4.append(b4/block)
        done += block
        if time.time()-t0 > time_budget:
            break
    return s, M, np.array(blocks_m2), np.array(blocks_m4)
'''
open('/work/wolff.py','w').write(code)
import sys, time
sys.path.insert(0,'/work')
import importlib, wolff
importlib.reload(wolff)
t=time.time()
s,M,b2,b4 = wolff.run(2.3, 1, 20000, 1000)
dt=time.time()-t
print('time',dt, 'clusters/s', 20000/dt)
U = 1-b4.mean()/(3*b2.mean()**2)
print('U',U, 'm2',b2.mean(), 'check M', M, sum(s))
