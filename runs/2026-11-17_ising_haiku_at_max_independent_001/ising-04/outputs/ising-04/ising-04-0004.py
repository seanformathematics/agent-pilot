import sys, time
sys.path.insert(0, '/work')
import numpy as np
from swlib import *

t0 = time.time()
N = 16

# ---- (a) Burn-in check: average m^2 per sweep from a random start (SW), 256 replicas
for T in [1.5, 2.3, 2.8]:
    rng = np.random.default_rng(777)
    s = init_spins(256, N, rng, 'random')
    m2_hist = []
    for t in range(120):
        s = sw_step(s, T, rng)
        m = s.reshape(256, -1).sum(axis=1)/(N*N)
        m2_hist.append(np.mean(m*m))
    m2_hist = np.array(m2_hist)
    print(f"burn-in T={T}: <m2> at sweeps 0,1,2,5,10,20,40,80,119 ->",
          np.round(m2_hist[[0,1,2,5,10,20,40,80,119]], 4), " mean(60..119)=", round(m2_hist[60:].mean(),4), flush=True)

# ---- (b) Metropolis cross-check at N=16 (checkerboard, vectorized over replicas)
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
            flip = acc & mk
            s = np.where(flip, -s, s).astype(np.int8)
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

tm = time.time()
for T in [2.3]:
    S2m, S4m = metro_run(T, R=256, nsweep=6000, nburn=500, seed=4242)
    Um, errm = jackknife_U(S2m, S4m, 6000)
    print(f"Metropolis N=16 T={T}: U={Um:.5f} +- {errm:.5f}  (time {time.time()-tm:.1f}s)", flush=True)
ts = time.time()
S1, S2, S4, _ = run(N, 2.3, R=256, nsweep=6000, nburn=300, seed=5151)
Us, errs = jackknife_U(S2, S4, 6000)
print(f"SW        N=16 T=2.3: U={Us:.5f} +- {errs:.5f}  (time {time.time()-ts:.1f}s)", flush=True)

# ---- (c) Honest-error check: scatter of U across independent seeds vs mean jackknife error (T=2.7)
Us_list = []; errs_list = []
for k in range(16):
    S1, S2, S4, _ = run(N, 2.7, R=64, nsweep=1000, nburn=200, seed=9000+k)
    U, e = jackknife_U(S2, S4, 1000)
    Us_list.append(U); errs_list.append(e)
Us_list = np.array(Us_list)
print(f"seed-scatter T=2.7 (16 seeds x 64 reps x 1000 sweeps): mean U={Us_list.mean():.5f}, "
      f"std across seeds={Us_list.std(ddof=1):.5f}, mean jackknife err={np.mean(errs_list):.5f}", flush=True)
print("total time", time.time()-t0)
