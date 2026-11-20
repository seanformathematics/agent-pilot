import random, math, time, os
os.makedirs("/work", exist_ok=True)
code = r'''
import random, math, time, sys
import numpy as np

def make_nbr(N):
    nb = []
    for i in range(N):
        for j in range(N):
            nb.append([((i+1)%N)*N+j, ((i-1)%N)*N+j, i*N+(j+1)%N, i*N+(j-1)%N])
    return nb

def run(N, T, nsteps, seed, nburn, s=None, rng=None):
    rng = random.Random(seed)
    nb = make_nbr(N)
    V = N*N
    if s is None:
        s = [1 if rng.random()<0.5 else -1 for _ in range(V)]
    M = sum(s)
    padd = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    rint = rng.randrange
    def cluster_step():
        nonlocal M
        seed_site = rint(V)
        sp = s[seed_site]
        s[seed_site] = -sp
        stack = [seed_site]
        size = 1
        while stack:
            x = stack.pop()
            for y in nb[x]:
                if s[y] == sp and rnd() < padd:
                    s[y] = -sp
                    stack.append(y)
                    size += 1
        M -= 2*sp*size
        return size
    for _ in range(nburn):
        cluster_step()
    m2 = np.empty(nsteps); m4 = np.empty(nsteps)
    for k in range(nsteps):
        cluster_step()
        m = M/V
        m2[k] = m*m; m4[k] = m**4
    return m2, m4, s
'''
open("/work/wolff.py","w").write(code)
import sys; sys.path.insert(0,"/work")
from wolff import run
import numpy as np
t=time.time()
m2,m4,s = run(16, 2.269, 20000, 1, 2000)
dt=time.time()-t
print("time per 20000 clusters", dt)
U = 1-np.mean(m4)/(3*np.mean(m2)**2)
print("U", U, "<m2>", np.mean(m2))
# autocorrelation of m2
x = m2-m2.mean()
ac = [np.dot(x[:-k],x[k:])/np.dot(x,x) for k in [1,2,5,10,20]]
print("ac", ac)
