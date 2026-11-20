import random, math, time, os
os.makedirs("/work", exist_ok=True)
code = r'''
import random, math, sys, time
import numpy as np

def run(N, T, seed, nsteps, nburn):
    rng = random.Random(seed)
    V = N*N
    nb = [0]*(4*V)
    for x in range(N):
        for y in range(N):
            i = x*N+y
            nb[4*i+0] = ((x+1)%N)*N+y
            nb[4*i+1] = ((x-1)%N)*N+y
            nb[4*i+2] = x*N+(y+1)%N
            nb[4*i+3] = x*N+(y-1)%N
    s = [1]*V
    padd = 1.0 - math.exp(-2.0/T)
    rnd = rng.random
    M = V
    def step(s, M):
        i0 = int(rnd()*V)
        sd = s[i0]
        stack = [i0]
        s[i0] = -sd
        cl = [i0]
        while stack:
            i = stack.pop()
            for k in range(4):
                j = nb[4*i+k]
                if s[j] == sd and rnd() < padd:
                    s[j] = -sd
                    stack.append(j)
                    cl.append(j)
        c = len(cl)
        M = M - 2*sd*c
        return M, c
    for _ in range(nburn):
        M, c = step(s, M)
    out = np.empty(nsteps); outc = np.empty(nsteps)
    for t in range(nsteps):
        M, c = step(s, M)
        out[t] = M
        outc[t] = c
    return out, outc
'''
open("/work/wolff.py","w").write(code)
import sys; sys.path.insert(0,"/work")
from wolff import run
t=time.time()
M,c = run(16, 2.3, 1, 20000, 1000)
print("time per 20000 clusters", time.time()-t, "mean cluster", c.mean())
import numpy as np
m=np.abs(M)/256
print("U", 1-np.mean(m**4)/(3*np.mean(m**2)**2))
