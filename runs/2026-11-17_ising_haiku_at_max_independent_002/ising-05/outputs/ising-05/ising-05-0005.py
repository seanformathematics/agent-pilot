
import os, sys, time, math
import numpy as np
os.chdir('/work'); sys.path.insert(0, '/work')
import wolff_py as W

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
NCH = 4
STEPS_PER_CHAIN = 500000
BL = 100
NTHERM = 10000
BUDGET = 400.0
nbrs = W.make_nbrs(16)
os.makedirs('/work/results', exist_ok=True)

def fname(T):
    return f'/work/results/T{T:.2f}.npz'

t0 = time.time()
for i, T in enumerate(Ts):
    if os.path.exists(fname(T)):
        continue
    if time.time() - t0 > BUDGET:
        print('budget reached before T=', T)
        break
    t1 = time.time()
    S2s, S4s, seeds = [], [], []
    nfl = 0
    for c in range(NCH):
        seed = 1000*(i+1) + c + 1
        seeds.append(seed)
        S2, S4, nf = W.wolff_run(16, T, seed, NTHERM, STEPS_PER_CHAIN//BL, BL, nbrs)
        S2s.append(S2); S4s.append(S4); nfl += nf
    S2 = np.concatenate(S2s); S4 = np.concatenate(S4s)
    np.savez(fname(T), S2=S2, S4=S4, seeds=np.array(seeds), nflip=nfl)
    U, err = W.jk_U(S2, S4, BL, J=100)
    print(f'T={T}: U={U:.5f} jk_err={err:.5f}  sign_flips={nfl}  time={time.time()-t1:.1f}s', flush=True)
print('total elapsed', round(time.time()-t0, 1))
