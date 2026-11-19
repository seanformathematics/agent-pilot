
import os, textwrap, time
os.makedirs("/work", exist_ok=True)
code = r'''
import random, math
import numpy as np

def make_nbr(N):
    nbr = []
    for y in range(N):
        for x in range(N):
            nbr.append((y*N + (x+1) % N,
                        y*N + (x-1) % N,
                        ((y+1) % N)*N + x,
                        ((y-1) % N)*N + x))
    return nbr

def wolff_run(N, T, seed, n_therm, n_meas, init="hot"):
    """Single-cluster Wolff updates.  Returns (M_series int16 after EVERY cluster flip during
    the measurement phase, mean cluster size during measurement, M_series during thermalisation)."""
    rng = random.Random(seed)
    rnd = rng.random
    n = N*N
    p = 1.0 - math.exp(-2.0/T)
    nbr = make_nbr(N)
    if init == "hot":
        s = [1 if rnd() < 0.5 else -1 for _ in range(n)]
    elif init == "cold":
        s = [1]*n
    else:
        raise ValueError(init)
    M = sum(s)
    th = [0]*n_therm
    out = [0]*n_meas
    total_c = 0
    for step in range(n_therm + n_meas):
        i0 = int(rnd()*n)
        sv = s[i0]
        nsv = -sv
        s[i0] = nsv
        stack = [i0]
        pop = stack.pop
        push = stack.append
        c = 1
        while stack:
            i = pop()
            for j in nbr[i]:
                if s[j] == sv and rnd() < p:
                    s[j] = nsv
                    push(j)
                    c += 1
        M += 2*nsv*c
        if step < n_therm:
            th[step] = M
        else:
            out[step-n_therm] = M
            total_c += c
    return (np.array(out, dtype=np.int16), total_c/max(n_meas, 1), np.array(th, dtype=np.int16))
'''
open("/work/ising.py","w").write(code)

import sys
sys.path.insert(0,"/work")
import numpy as np
from ising import wolff_run

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N=16
for T in Ts:
    t=time.time()
    M, cbar, th = wolff_run(N, T, seed=1000+int(T*100), n_therm=2000, n_meas=20000, init="hot")
    dt=time.time()-t
    m=M/256.0
    m2=np.mean(m**2); m4=np.mean(m**4)
    U=1-m4/(3*m2**2)
    print(f"T={T:.1f} cbar={cbar:7.1f}  time/flip={dt/22000*1e6:6.1f}us  <m2>={m2:.4f} U={U:.4f}  flips/sweep={256/cbar:.2f}")
