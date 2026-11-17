import sys, time, glob
sys.path.insert(0, '/work')
import numpy as np
from swlib import *

# Reload production data and recompute U and jackknife error at full precision
Ts_all = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
prod = {}
for T in Ts_all:
    d = np.load(f'/work/results/prod_T{T:.2f}.npz')
    U, err = jackknife_U(d['S2'], d['S4'], int(d['NSW']))
    prod[T] = (U, err, int(d['R'])*int(d['NSW']))
    print(f"T={T:.2f}  U={U:.7f}  U_err={err:.3e}  n_meas={int(d['R'])*int(d['NSW'])}  seed={int(d['seed'])}")

# Independent Metropolis cross-check (checkerboard, vectorized over replicas) at several T
N = 16
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
masks = [((ii+jj) % 2 == 0), ((ii+jj) % 2 == 1)]
def metro_run(T, R, nsweep, nburn, seed):
    rng = np.random.default_rng(seed)
    s = init_spins(R, N, rng, 'random')
    ptab = np.array([min(1.0, np.exp(-dE/T)) for dE in (-8, -4, 0, 4, 8)])
    def sweep(s):
        for mk in masks:
            nb = np.roll(s,1,axis=1)+np.roll(s,-1,axis=1)+np.roll(s,1,axis=2)+np.roll(s,-1,axis=2)
            dE = 2*s*nb
            k = (dE + 8)//4
            acc = rng.random(s.shape) < ptab[k]
            s = np.where(acc & mk, -s, s).astype(np.int8)
        return s
    for _ in range(nburn):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R)
    for _ in range(nsweep):
        s = sweep(s)
        m = s.reshape(R,-1).sum(axis=1)/(N*N)
        m2 = m*m
        S2 += m2; S4 += m2*m2
    return S2, S4

t0 = time.time()
for T, seed in [(2.0, 6001), (2.2, 6002), (2.4, 6003), (2.6, 6004), (2.8, 6005)]:
    S2m, S4m = metro_run(T, R=256, nsweep=8000, nburn=500, seed=seed)
    Um, em = jackknife_U(S2m, S4m, 8000)
    Us, es, _ = prod[T]
    z = (Um - Us)/np.sqrt(em**2 + es**2)
    print(f"check T={T}: Metropolis U={Um:.5f}+-{em:.5f} | SW prod U={Us:.5f}+-{es:.5f} | z={z:+.2f}", flush=True)
print("cross-check time", round(time.time()-t0,1))
