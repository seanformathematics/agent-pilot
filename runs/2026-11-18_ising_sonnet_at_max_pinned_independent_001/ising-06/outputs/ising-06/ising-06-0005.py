
code = r'''
import numpy as np
import sw

def run_T(N, T, R, n_eq, n_meas, B, seed, start="hot", eq_trace=False):
    """Run R independent replicas: n_eq SW equilibration updates, then n_meas measured updates.
    Returns block means (B updates per block) of [m2, m4, i2, i4, e] with shape (5, R, nblocks)."""
    b = sw.SWBatch(N, R, T, seed, start)
    tr = []
    for t in range(n_eq):
        b.step()
        if eq_trace:
            m, e = b.observables()
            tr.append((e.mean(), (m**2).mean()))
    nb = n_meas // B
    out = np.zeros((5, R, nb))
    n2 = float(N**2); n4 = float(N**4); n8 = float(N**8)
    for ib in range(nb):
        acc = np.zeros((5, R))
        for k in range(B):
            S2, S4 = b.step()
            m, e = b.observables()
            m2 = m * m
            acc[0] += m2
            acc[1] += m2 * m2
            acc[2] += S2 / n4
            acc[3] += (3.0 * S2 * S2 - 2.0 * S4) / n8
            acc[4] += e
        out[:, :, ib] = acc / B
    return out, np.array(tr)

def U_from(m2, m4):
    return 1.0 - m4 / (3.0 * m2 * m2)

def jack_U(X, i2, i4):
    """X: (5,R,nb) block means. Jackknife over replicas for U built from rows (i2,i4). Returns U, err."""
    A = X[i2].mean(axis=1)   # per-replica mean of m2   (equal block counts -> equal weights)
    Bq = X[i4].mean(axis=1)
    R = A.shape[0]
    sA, sB = A.sum(), Bq.sum()
    U = U_from(sA / R, sB / R)
    Uj = U_from((sA - A) / (R - 1), (sB - Bq) / (R - 1))
    err = np.sqrt((R - 1) / R * np.sum((Uj - Uj.mean())**2))
    return U, err

def jack_mean(x):
    R = x.shape[0]
    return x.mean(), x.std(ddof=1) / np.sqrt(R)
'''
open('/work/run.py','w').write(code)

import sys, time
sys.path.insert(0,'/work')
import numpy as np, importlib, sw, run, exact_torus as ex
importlib.reload(run)

Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
N=16
t0=time.time()
print("T   | e_exact      e_hot(±)        e_cold(±)     | U_plain(±)       U_improved(±)   | hot-cold dU")
res = {}
for iT,T in enumerate(Ts):
    eexact = ex.energy_per_spin_exact(T, N)
    Xh,_ = run.run_T(N, T, R=128, n_eq=100, n_meas=1000, B=50, seed=1000+iT, start="hot")
    Xc,_ = run.run_T(N, T, R=128, n_eq=100, n_meas=1000, B=50, seed=2000+iT, start="cold")
    eh, ehe = run.jack_mean(Xh[4].mean(axis=1))
    ec, ece = run.jack_mean(Xc[4].mean(axis=1))
    Xall = np.concatenate([Xh, Xc], axis=1)
    Up, Upe = run.jack_U(Xall, 0, 1)
    Ui, Uie = run.jack_U(Xall, 2, 3)
    Uh,_ = run.jack_U(Xh, 2, 3); Uc,_ = run.jack_U(Xc, 2, 3)
    print(f"{T:.1f} | {eexact:.5f}  {eh:.5f}({ehe:.5f})  {ec:.5f}({ece:.5f}) | {Up:.4f}({Upe:.4f})  {Ui:.4f}({Uie:.4f}) | {Uh-Uc:+.4f}", flush=True)
    res[T]=(Up,Upe,Ui,Uie)
print("elapsed", time.time()-t0)
