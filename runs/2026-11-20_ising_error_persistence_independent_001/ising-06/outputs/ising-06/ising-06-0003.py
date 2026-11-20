
import random, math, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import random, math, sys, time
def run(N, T, nsweeps_meas, seed, therm_clusters, out_path=None):
    rng = random.Random(seed)
    L = N*N
    nb = [[0]*4 for _ in range(L)]
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[i][0] = ((x+1)%N)*N+y
            nb[i][1] = ((x-1)%N)*N+y
            nb[i][2] = x*N+(y+1)%N
            nb[i][3] = x*N+(y-1)%N
    s = [1]*L
    padd = 1.0-math.exp(-2.0/T)
    rand = rng.random; randint = rng.randrange
    M = L  # total spin sum
    def cluster_flip(M):
        i0 = randint(L)
        s0 = s[i0]
        s[i0] = -s0
        stack = [i0]
        size = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if s[j] == s0 and rand() < padd:
                    s[j] = -s0
                    stack.append(j)
                    size += 1
        return size*(-2*s0)  # change in sum
    S = sum(s)
    for _ in range(therm_clusters):
        S += cluster_flip(S)
    res = []
    for _ in range(nsweeps_meas):
        S += cluster_flip(S)
        m = S/L
        res.append(m)
    return res
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time()
r=run(16,2.269,20000,1,1000)
print('time per 20000 clusters',time.time()-t)
import statistics
print(statistics.mean(abs(x) for x in r))
