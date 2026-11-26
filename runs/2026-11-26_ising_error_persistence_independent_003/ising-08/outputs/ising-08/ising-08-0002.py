
import random, math, time, os, json
os.makedirs('/work',exist_ok=True)
src = r'''
import random, math, sys, json, time
def run(N, T, seed, n_eq, n_meas, nbin_out=None):
    rng = random.Random(seed)
    V = N*N
    nb = [[0]*4 for _ in range(V)]
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[i] = [((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N]
    s = [1]*V
    p = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    rr = rng.randrange
    def wolff():
        i0 = rr(V)
        c = s[i0]
        s[i0] = -c
        stack = [i0]
        size = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j] == c and rnd() < p:
                    s[j] = -c
                    stack.append(j)
                    size += 1
        return size
    for _ in range(n_eq):
        wolff()
    m2 = [0.0]*n_meas; m4=[0.0]*n_meas
    for k in range(n_meas):
        wolff()
        M = sum(s)
        m = M / V
        m2v = m*m
        m2[k] = m2v
        m4[k] = m2v*m2v
    return m2, m4
'''
open('/work/ising.py','w').write(src)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time(); m2,m4=run(16,2.3,1,500,2000); print('time',time.time()-t)
import statistics
print(sum(m2)/len(m2), sum(m4)/len(m4))
