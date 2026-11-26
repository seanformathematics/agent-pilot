
import random, math, time, os
os.makedirs('/work', exist_ok=True)
src = r'''
import random, math
def make_nbr(N):
    nb = [None]*(N*N)
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[i] = (((x+1)%N)*N+y, ((x-1)%N)*N+y, x*N+(y+1)%N, x*N+(y-1)%N)
    return nb
def run(N, T, nsweep_eq, nmeas, seed, meas_every=1):
    rng = random.Random(seed)
    rnd = rng.random
    nb = make_nbr(N)
    V = N*N
    s = [1]*V
    p = 1.0 - math.exp(-2.0/T)
    M = []
    def wolff():
        i0 = int(rnd()*V)
        c = s[i0]
        s[i0] = -c
        stack = [i0]
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j] == c and rnd() < p:
                    s[j] = -c
                    stack.append(j)
    # equilibrate
    for _ in range(nsweep_eq):
        wolff()
    out = []
    k = 0
    for _ in range(nmeas):
        wolff()
        k += 1
        if k % meas_every == 0:
            m = sum(s)/V
            out.append(m)
    return out
'''
open('/work/ising.py','w').write(src)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time()
out = run(16, 2.27, 2000, 20000, 1)
print("time for 22000 clusters", time.time()-t)
import statistics
ms=[abs(x) for x in out]
print(statistics.mean(ms))
