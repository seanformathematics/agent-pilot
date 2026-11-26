
import numpy as np, time, os, json
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np
def run(T, N=16, R=1000, nsweep=20000, ntherm=2000, seed=1):
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1, 1], size=(R, 1, 1)).astype(np.int8)
    s = np.broadcast_to(sign, (R, N, N)).copy()
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == c) for c in (0, 1)]
    P = np.zeros((2, 5))
    for si, s0 in enumerate((-1, 1)):
        for hi, h in enumerate((-4, -2, 0, 2, 4)):
            P[si, hi] = min(1.0, np.exp(-2*s0*h/T))
    def sweep(s):
        for mk in masks:
            h = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            si = (s > 0).astype(np.int8)
            hi = ((h + 4)//2).astype(np.int8)
            p = P[si, hi]
            u = rng.random((R, N, N))
            flip = (u < p) & mk
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(ntherm):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R); n = 0
    blk2 = []; blk4 = []  # per-1000-sweep block sums for drift check
    b2 = np.zeros(R); b4 = np.zeros(R); bn = 0
    for k in range(nsweep):
        s = sweep(s)
        m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
        m2 = m*m; m4 = m2*m2
        S2 += m2; S4 += m4; n += 1
        b2 += m2; b4 += m4; bn += 1
        if bn == 2000:
            blk2.append(b2/bn); blk4.append(b4/bn); b2 = np.zeros(R); b4 = np.zeros(R); bn = 0
    return S2/n, S4/n, np.array(blk2), np.array(blk4)

def binder_jk(A2, A4, nb=None):
    # A2, A4: per-replica time averages (R,). jackknife over replicas.
    R = len(A2)
    M2 = A2.mean(); M4 = A4.mean()
    U = 1 - M4/(3*M2**2)
    # leave-one-out
    tot2 = A2.sum(); tot4 = A4.sum()
    l2 = (tot2 - A2)/(R-1); l4 = (tot4 - A4)/(R-1)
    Uj = 1 - l4/(3*l2**2)
    err = np.sqrt((R-1)/R * np.sum((Uj - Uj.mean())**2))
    return U, err
'''
open('/work/ising.py','w').write(src)
import sys; sys.path.insert(0,'/work')
from ising import run, binder_jk
out = {}
for T, seed in [(1.9, 101), (2.0, 102)]:
    t0 = time.time()
    A2, A4, B2, B4 = run(T, R=1000, nsweep=20000, ntherm=2000, seed=seed)
    U, err = binder_jk(A2, A4)
    Ub = [1 - b4.mean()/(3*b2.mean()**2) for b2, b4 in zip(B2, B4)]
    np.savez(f'/work/res_T{T}.npz', A2=A2, A4=A4, B2=B2, B4=B4)
    print(T, "U=%.5f err=%.5f" % (U, err), "blocks:", np.round(Ub, 4), "time %.1f" % (time.time()-t0), flush=True)
