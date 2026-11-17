import random, math, time, os, json
os.makedirs('/work', exist_ok=True)
code = r'''
import random, math, time, sys, json
N=16; V=N*N
nb=[]
for i in range(V):
    x,y=i%N,i//N
    nb.append([((x+1)%N)+y*N, ((x-1)%N)+y*N, x+((y+1)%N)*N, x+((y-1)%N)*N])
def run(T, ncl, neq, seed, nbin_target=None):
    rng=random.Random(seed)
    p=1-math.exp(-2.0/T)
    s=[1]*V
    rnd=rng.random; rr=rng.randrange
    M=V
    m=V  # sum of spins
    def flip_cluster(start):
        nonlocal m
        sv=s[start]
        s[start]=-sv
        stack=[start]; cnt=1
        while stack:
            i=stack.pop()
            for j in nb[i]:
                if s[j]==sv and rnd()<p:
                    s[j]=-sv; stack.append(j); cnt+=1
        m += -2*sv*cnt
    for _ in range(neq):
        flip_cluster(rr(V))
    out=[]
    for _ in range(ncl):
        flip_cluster(rr(V))
        # measure after each cluster
        out.append(m)
    return out
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time(); o=run(2.27,20000,1000,1); print('time',time.time()-t)
import statistics
print(statistics.mean(abs(x) for x in o)/256)
