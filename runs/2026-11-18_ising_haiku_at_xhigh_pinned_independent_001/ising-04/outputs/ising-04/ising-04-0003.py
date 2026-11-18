import os
os.makedirs('/work', exist_ok=True)
mod = r'''
import numpy as np
N = 16
_ii, _jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
MASKS = [((_ii + _jj) % 2 == p) for p in (0, 1)]

def make_thr(T):
    return np.array([min(1.0, np.exp(-2.0 * x / T)) for x in range(-4, 5)], dtype=np.float32)

def sweep(S, thr, rng):
    for p in (0, 1):
        nn = np.roll(S, 1, 1) + np.roll(S, -1, 1)
        nn += np.roll(S, 1, 2)
        nn += np.roll(S, -1, 2)
        x = S * nn
        t = thr[x + 4]
        acc = rng.random(S.shape, dtype=np.float32) < t
        acc &= MASKS[p]
        S *= np.where(acc, np.int8(-1), np.int8(1))
    return S

def run(T, R, n_equil, n_meas, seed, start='random', S=None, rng=None, every=1):
    """Returns per-replica sums (float64) of m^2, m^4, and replica-averaged per-sweep series."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if S is None:
        if start == 'random':
            S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
        else:
            S = np.ones((R, N, N), dtype=np.int8)
    thr = make_thr(T)
    for _ in range(n_equil):
        sweep(S, thr, rng)
    A = np.zeros(R); B = np.zeros(R); Mabs = np.zeros(R); E = np.zeros(R)
    ser_m2 = []; ser_m4 = []; ser_m1 = []; ser_e = []
    K = 0
    for t in range(n_meas):
        sweep(S, thr, rng)
        if t % every == 0:
            m = S.reshape(R, -1).sum(axis=1, dtype=np.int64) / (N * N)
            m2 = m * m
            m4 = m2 * m2
            A += m2; B += m4; Mabs += np.abs(m)
            ser_m2.append(m2.mean()); ser_m4.append(m4.mean()); ser_m1.append(np.abs(m).mean())
            K += 1
    return dict(S=S, A=A, B=B, Mabs=Mabs, K=K,
                ser_m2=np.array(ser_m2), ser_m4=np.array(ser_m4), ser_m1=np.array(ser_m1))

def binder_jk(A, B, K):
    """Binder U from per-replica sums; delete-one-replica jackknife error."""
    R = len(A)
    SA = A.sum(); SB = B.sum()
    U = 1.0 - (SB * R * K) / (3.0 * SA * SA)
    Ui = 1.0 - ((SB - B) * (R - 1) * K) / (3.0 * (SA - A) ** 2)
    Ubar = Ui.mean()
    err = np.sqrt((R - 1) / R * np.sum((Ui - Ubar) ** 2))
    return U, err

def tau_int(x, maxlag=None):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if maxlag is None:
        maxlag = n // 4
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    # Sokal automatic window
    tau = 0.5
    for w in range(1, maxlag):
        tau += ac[w]
        if w >= 6 * tau:
            break
    return tau
'''
open('/work/ising.py', 'w').write(mod)
import sys
sys.path.insert(0, '/work')
import importlib, time
import numpy as np
import ising
importlib.reload(ising)

for T in [2.27, 1.5]:
    t0 = time.time()
    out = ising.run(T, R=1024, n_equil=1500, n_meas=1500, seed=11, start='random')
    U, err = ising.binder_jk(out['A'], out['B'], out['K'])
    tau2 = ising.tau_int(out['ser_m2'])
    tau4 = ising.tau_int(out['ser_m4'])
    print(f"T={T} U={U:.5f} err={err:.5f} tau(m2)={tau2:.1f} tau(m4)={tau4:.1f} time={time.time()-t0:.1f}s")
    # block-trend check
    q = out['ser_m2'].reshape(5, -1).mean(axis=1)
    print("  m2 by 300-sweep blocks:", np.round(q, 4))
    print("  <|m|> =", out['Mabs'].sum()/(1024*out['K']))
