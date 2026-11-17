
import os
os.makedirs('/work/res', exist_ok=True)
src = r'''
import numpy as np, time

def make_state(R, N, rng):
    return np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)

def sw_step(S, T, rng, N):
    R = S.shape[0]
    NN = N * N
    p = 1.0 - np.exp(-2.0 / T)
    right = S == np.roll(S, -1, axis=2)
    down = S == np.roll(S, -1, axis=1)
    aR = right & (rng.random((R, N, N)) < p)
    aD = down & (rng.random((R, N, N)) < p)
    aRl = np.roll(aR, 1, axis=2)
    aDu = np.roll(aD, 1, axis=1)
    BIG = np.int32(NN + 5)
    L = np.broadcast_to(np.arange(NN, dtype=np.int32).reshape(1, N, N), (R, N, N)).copy()
    while True:
        c1 = np.where(aR, np.roll(L, -1, axis=2), BIG)
        c2 = np.where(aRl, np.roll(L, 1, axis=2), BIG)
        c3 = np.where(aD, np.roll(L, -1, axis=1), BIG)
        c4 = np.where(aDu, np.roll(L, 1, axis=1), BIG)
        newL = np.minimum(np.minimum(L, c1), np.minimum(np.minimum(c2, c3), c4))
        nf = newL.reshape(R, NN)
        newL = np.take_along_axis(nf, nf, axis=1).reshape(R, N, N)
        if np.array_equal(newL, L):
            break
        L = newL
    Lf = L.reshape(R, NN)
    coin = rng.random((R, NN)) < 0.5
    flip = np.take_along_axis(coin, Lf, axis=1).reshape(R, N, N)
    S = np.where(flip, -S, S).astype(np.int8)
    return S

def measure(S, N):
    R = S.shape[0]
    NN = N * N
    m = S.reshape(R, NN).mean(axis=1, dtype=np.float64)
    E = -((S * np.roll(S, -1, axis=2)).reshape(R, NN).sum(axis=1, dtype=np.int64)
          + (S * np.roll(S, -1, axis=1)).reshape(R, NN).sum(axis=1, dtype=np.int64))
    return m, E / NN

def run(T, N, R, n_burn, n_meas, seed, every=1):
    rng = np.random.default_rng(seed)
    S = make_state(R, N, rng)
    for _ in range(n_burn):
        S = sw_step(S, T, rng, N)
    m2s = np.zeros(R); m4s = np.zeros(R); ams = np.zeros(R); es = np.zeros(R); cnt = 0
    for k in range(n_meas):
        S = sw_step(S, T, rng, N)
        if k % every == 0:
            m, e = measure(S, N)
            m2 = m * m
            m2s += m2; m4s += m2 * m2; ams += np.abs(m); es += e; cnt += 1
    return dict(m2=m2s, m4=m4s, am=ams, e=es, cnt=np.full(R, cnt))
'''
open('/work/sw.py', 'w').write(src)
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import *
for T in [2.269, 1.5, 2.8]:
    rng = np.random.default_rng(0)
    S = make_state(1024, 16, rng)
    t0 = time.time()
    for _ in range(20):
        S = sw_step(S, T, rng, 16)
    dt = (time.time() - t0) / 20
    print(T, 'time per SW step (R=1024):', round(dt, 4), 's')
