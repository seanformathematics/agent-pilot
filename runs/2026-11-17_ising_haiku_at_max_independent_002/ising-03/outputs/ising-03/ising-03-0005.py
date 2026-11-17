prod = r'''
import numpy as np, time
from sw import SW

def run_T(T, seed, R=2000, burn=200, nmeas=1000, blk=20, N=16):
    rng = np.random.default_rng(seed)
    s = SW(N, R, T, rng, init="random")
    for _ in range(burn):
        s.sweep()
    nb = nmeas // blk
    S2 = np.zeros((R, nb)); S4 = np.zeros((R, nb))
    for b in range(nb):
        acc2 = np.zeros(R); acc4 = np.zeros(R)
        for t in range(blk):
            s.sweep()
            m = s.m()
            m2 = m*m
            acc2 += m2; acc4 += m2*m2
        S2[:, b] = acc2; S4[:, b] = acc4
    np.savez(f"/work/data/prod_T{T:.2f}_seed{seed}.npz", S2=S2, S4=S4, R=R, burn=burn,
             nmeas=nmeas, blk=blk, T=T, seed=seed)
    return S2, S4

def U_jack(S2, S4, nmeas, G=100):
    R = S2.shape[0]
    tot2 = S2.sum(axis=1); tot4 = S4.sum(axis=1)
    B2 = tot2.sum()/(R*nmeas); B4 = tot4.sum()/(R*nmeas)
    U = 1 - B4/(3*B2**2)
    groups = np.array_split(np.arange(R), G)
    Ug = []
    for g in groups:
        keep = np.ones(R, bool); keep[g] = False
        n = keep.sum()*nmeas
        b2 = tot2[keep].sum()/n; b4 = tot4[keep].sum()/n
        Ug.append(1 - b4/(3*b2**2))
    Ug = np.array(Ug)
    err = np.sqrt((G-1)/G*np.sum((Ug-Ug.mean())**2))
    return U, err
'''
with open("/work/prod.py", "w") as f:
    f.write(prod)

import sys, time
sys.path.insert(0, "/work")
import numpy as np
from prod import run_T, U_jack

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
t0 = time.time()
for k, T in enumerate(Ts):
    seed = 777001 + k
    S2, S4 = run_T(T, seed)
    U, err = U_jack(S2, S4, 1000)
    print(f"T={T:.2f} seed={seed} U={U:.6f} err={err:.3e} n={S2.size*20} elapsed={time.time()-t0:.0f}s", flush=True)
