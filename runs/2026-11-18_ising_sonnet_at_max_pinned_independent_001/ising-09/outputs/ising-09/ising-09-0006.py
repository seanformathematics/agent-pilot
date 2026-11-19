
import sys, time
sys.path.insert(0, "/work")
code = r'''
import numpy as np
from sw import SW

def ratio_U(a2, a4):
    return 1.0 - a4/(3.0*a2**2)

def jackknife_U(x2, x4):
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
    Returns acc (5, R, nb): block means of [m2, m4, m2_imp, m4_imp, e]."""
    sim = SW(N, R, seed, start)
    for _ in range(E):
        sim.sweep(T)
    nb = S // L
    acc = np.zeros((5, R, nb))
    for b in range(nb):
        for k in range(L):
            imp = sim.sweep(T, measure_improved=improved)
            s = sim.s
            m = s.reshape(R, -1).sum(axis=1, dtype=np.int64)/(N*N)
            m2 = m*m
            acc[0, :, b] += m2
            acc[1, :, b] += m2*m2
            if improved:
                acc[2, :, b] += imp[0]
                acc[3, :, b] += imp[1]
            en = -(s*np.roll(s, -1, axis=2) + s*np.roll(s, -1, axis=1)).sum(axis=(1, 2), dtype=np.int64)/(N*N)
            acc[4, :, b] += en
    return acc/L

def summarize(acc):
    out = {}
    four, R, nb = acc.shape[0], acc.shape[1], acc.shape[2]
    for name, (i2, i4) in {"plain": (0, 1), "improved": (2, 3)}.items():
        rep2 = acc[i2].mean(axis=1); rep4 = acc[i4].mean(axis=1)
        U, err_rep, Ujk = jackknife_U(rep2, rep4)
        U_b, err_blk, _ = jackknife_U(acc[i2].ravel(), acc[i4].ravel())
        out[name] = dict(U=U, err_rep=err_rep, err_blk=err_blk, Ujk=Ujk, m2=rep2.mean(), m4=rep4.mean())
    rep_e = acc[4].mean(axis=1)
    out["e"] = (rep_e.mean(), rep_e.std(ddof=1)/np.sqrt(R))
    return out
'''
open("/work/run.py","w").write(code)

import numpy as np, time
from run import *
from exact import energy_exact
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
t0=time.time()
pilot = {}
print(" T    | e_MC (err)  e_exact  z_e | plain U (err_rep, err_blk) | improved U (err_rep) | var ratio")
for T in Ts:
    acc = run_T(16, T, R=500, E=300, S=400, L=20, seed=2024+int(round(T*100)))
    pilot[T] = acc
    r = summarize(acc)
    ee = energy_exact(16, T)
    z = (r['e'][0]-ee)/r['e'][1]
    print(f"{T:.1f} | {r['e'][0]:.5f} ({r['e'][1]:.5f}) {ee:.5f} {z:+.2f} | {r['plain']['U']:.5f} ({r['plain']['err_rep']:.5f}, {r['plain']['err_blk']:.5f}) | {r['improved']['U']:.5f} ({r['improved']['err_rep']:.5f}) | {(r['plain']['err_rep']/r['improved']['err_rep'])**2:.2f}")
np.savez("/work/pilot.npz", **{f"T{T}": pilot[T] for T in Ts})
print("time", time.time()-t0)
