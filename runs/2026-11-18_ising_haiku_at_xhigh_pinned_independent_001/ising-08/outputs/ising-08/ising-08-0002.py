
code = r'''
import numpy as np, sys, time, json

def run(seed, temps, R, equil, block, time_budget, out, hot=True):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    temps = np.asarray(temps, dtype=float)
    nT = len(temps)
    L = 16
    Rtot = nT * R
    temp_of = np.repeat(temps, R)
    beta = 1.0 / temp_of
    if hot:
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(Rtot, L, L))
    else:
        s = np.ones((Rtot, L, L), dtype=np.int8)
    ii, jj = np.indices((L, L))
    black = ((ii + jj) % 2 == 0)[None]
    white = ~black
    xs = np.array([-4, -2, 0, 2, 4])
    ptab = np.where(xs[None, :] > 0, np.exp(-2.0 * beta[:, None] * xs[None, :]), 1.0)
    ridx = np.arange(Rtot)[:, None, None]
    blk_m2, blk_m4, blk_am = [], [], []
    acc_m2 = np.zeros(Rtot); acc_m4 = np.zeros(Rtot); acc_am = np.zeros(Rtot)
    nsw = 0
    nb_done = 0
    while True:
        for mask in (black, white):
            h = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            x = s * h
            j = ((x + 4) >> 1).astype(np.intp)
            p = ptab[ridx, j]
            r = rng.random((Rtot, L, L), dtype=np.float32)
            flip = (r < p) & mask
            s = np.where(flip, -s, s).astype(np.int8)
        nsw += 1
        m = s.reshape(Rtot, -1).sum(axis=1, dtype=np.int64) / (L * L)
        m2 = m * m
        if nsw > equil:
            acc_m2 += m2; acc_m4 += m2 * m2; acc_am += np.abs(m)
        if nsw > equil and (nsw - equil) % block == 0:
            blk_m2.append(acc_m2.copy()); blk_m4.append(acc_m4.copy()); blk_am.append(acc_am.copy())
            acc_m2[:] = 0; acc_m4[:] = 0; acc_am[:] = 0
            nb_done += 1
            if time.time() - t0 > time_budget:
                break
        if time.time() - t0 > time_budget and nsw > equil:
            break
    np.savez(out, m2=np.array(blk_m2), m4=np.array(blk_m4), am=np.array(blk_am),
             temps=temps, R=R, equil=equil, block=block, seed=seed, nsweeps=nsw)
    return nsw, nb_done, time.time() - t0
'''
open('/work/sim.py', 'w').write(code) if __import__('os').path.isdir('/work') else None
import os
os.makedirs('/work', exist_ok=True)
open('/work/sim.py', 'w').write(code)
import sys
sys.path.insert(0, '/work')
from sim import run
temps = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
nsw, nb, dt = run(seed=1, temps=temps, R=32, equil=0, block=50, time_budget=40, out='/work/pilot.npz')
print("sweeps", nsw, "blocks", nb, "time", dt, "per sweep ms", 1000*dt/nsw)
