
import sys, time
sys.path.insert(0, "/work")
code = r'''
import numpy as np
from sw import SW

def ratio_U(a2, a4):
    return 1.0 - a4/(3.0*a2**2)

def jackknife_U(x2, x4):
    """x2, x4: per-unit means (equal weight units). Returns U, jackknife err, bias-corrected U."""
    x2 = np.asarray(x2, float); x4 = np.asarray(x4, float)
    n = len(x2)
    A2 = x2.mean(); A4 = x4.mean()
    U = ratio_U(A2, A4)
    A2_i = (x2.sum() - x2)/(n-1); A4_i = (x4.sum() - x4)/(n-1)
    Ui = ratio_U(A2_i, A4_i)
    var = (n-1)/n*np.sum((Ui - Ui.mean())**2)
    Ujk = n*U - (n-1)*Ui.mean()
    return U, np.sqrt(var), Ujk

def run_T(N, T, R, E, S, L, seed, start="hot", improved=True):
    """R replicas, E equilibration sweeps, S measurement sweeps per replica, blocks of L sweeps.
    Returns acc array (4, R, nb): block sums of [m2, m4, m2_imp, m4_imp]."""
    sim = SW(N, R, seed, start)
    for _ in range(E):
        sim.sweep(T)
    nb = S // L
    acc = np.zeros((4, R, nb))
    for b in range(nb):
        for k in range(L):
            imp = sim.sweep(T, measure_improved=improved)
            m = sim.mag()
            m2 = m*m
            acc[0, :, b] += m2
            acc[1, :, b] += m2*m2
            if improved:
                acc[2, :, b] += imp[0]
                acc[3, :, b] += imp[1]
    return acc/L   # block means

def summarize(acc):
    """acc: (4,R,nb) block means. Return dict of estimates using replica-level and block-level jackknife."""
    out = {}
    four, R, nb = acc.shape
    for name, (i2, i4) in {"plain": (0, 1), "improved": (2, 3)}.items():
        rep2 = acc[i2].mean(axis=1); rep4 = acc[i4].mean(axis=1)
        U, err_rep, Ujk = jackknife_U(rep2, rep4)
        blk2 = acc[i2].ravel(); blk4 = acc[i4].ravel()
        U_b, err_blk, _ = jackknife_U(blk2, blk4)
        out[name] = dict(U=U, err_rep=err_rep, err_blk=err_blk, Ujk=Ujk, m2=rep2.mean(), m4=rep4.mean())
    return out
'''
open("/work/run.py","w").write(code)

import numpy as np
from run import *

# exact enumeration for N=4
N=4
conf = np.arange(2**16)
bits = ((conf[:,None] >> np.arange(16)) & 1)*2 - 1
s = bits.reshape(-1,4,4)
E = -(s*np.roll(s,-1,axis=1) + s*np.roll(s,-1,axis=2)).sum(axis=(1,2))
M = s.sum(axis=(1,2))/16.0
def exact(T):
    w = np.exp(-(E-E.min())/T); Z = w.sum()
    m2 = (w*M**2).sum()/Z; m4 = (w*M**4).sum()/Z
    e = (w*E).sum()/Z/16
    return m2, m4, 1-m4/(3*m2**2), e

t0=time.time()
for T in [1.5, 2.0, 2.3, 2.8]:
    m2e, m4e, Ue, ee = exact(T)
    acc = run_T(4, T, R=500, E=100, S=1000, L=50, seed=12345+int(T*10))
    res = summarize(acc)
    print(f"T={T}: exact m2={m2e:.5f} m4={m4e:.5f} U={Ue:.5f}")
    for k,v in res.items():
        z_rep = (v['U']-Ue)/v['err_rep']
        print(f"   {k:9s}: U={v['U']:.5f} err_rep={v['err_rep']:.5f} err_blk={v['err_blk']:.5f}  z={z_rep:+.2f}  m2={v['m2']:.5f} m4={v['m4']:.5f}")
print("time", time.time()-t0)
