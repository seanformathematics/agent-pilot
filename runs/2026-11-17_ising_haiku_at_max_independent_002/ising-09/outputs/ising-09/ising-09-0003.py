lib = r'''
import numpy as np, math, time, json, os

def wolff_step(s, rng, p, N, R, ar):
    # bonds: open with prob p if the two spins are aligned (periodic boundaries)
    bR = (s == np.roll(s, -1, axis=1)) & (rng.random((N, N, R), dtype=np.float32) < p)
    bD = (s == np.roll(s, -1, axis=0)) & (rng.random((N, N, R), dtype=np.float32) < p)
    bL = np.roll(bR, 1, axis=1)
    bU = np.roll(bD, 1, axis=0)
    i0 = rng.integers(0, N, R)
    j0 = rng.integers(0, N, R)
    C = np.zeros((N, N, R), dtype=bool)
    C[i0, j0, ar] = True            # Wolff seed site per replica
    iters = 0
    while True:                     # flood fill = Wolff cluster through open bonds
        iters += 1
        new = (C | np.roll(C & bR, 1, axis=1) | np.roll(C & bL, -1, axis=1)
                 | np.roll(C & bD, 1, axis=0) | np.roll(C & bU, -1, axis=0))
        if np.array_equal(new, C):
            break
        C = new
    s = np.where(C, -s, s).astype(np.int8)
    return s, iters

def run_chain(N, R, T, nburn, nmeas, seed):
    rng = np.random.default_rng(seed)
    p = 1.0 - math.exp(-2.0 / T)
    ar = np.arange(R)
    s = np.where(rng.random((N, N, R)) < 0.5, 1, -1).astype(np.int8)   # random (infinite-T) start
    hist = []
    for t in range(nburn):
        s, it = wolff_step(s, rng, p, N, R, ar)
        m = s.sum(axis=(0, 1), dtype=np.int32) / (N * N)
        hist.append(float((m * m).mean()))
    m2 = np.empty((nmeas, R)); m4 = np.empty((nmeas, R)); iters_tot = 0
    t0 = time.time()
    for t in range(nmeas):
        s, it = wolff_step(s, rng, p, N, R, ar)
        iters_tot += it
        m = s.sum(axis=(0, 1), dtype=np.int32) / (N * N)
        m2[t] = m * m; m4[t] = m ** 4
        hist.append(float((m * m).mean()))
    el = time.time() - t0
    return m2.T.copy(), m4.T.copy(), np.array(hist), el, iters_tot / nmeas

def jack_U_rep(S2, S4, n):
    # replica-level jackknife: S2,S4 are per-replica sums over n samples
    R = S2.shape[0]
    tot2 = S2.sum(); tot4 = S4.sum()
    B = tot2 / (R * n); A = tot4 / (R * n)
    U = 1 - A / (3 * B * B)
    B_i = (tot2 - S2) / ((R - 1) * n); A_i = (tot4 - S4) / ((R - 1) * n)
    U_i = 1 - A_i / (3 * B_i ** 2)
    err = math.sqrt((R - 1) / R * np.sum((U_i - U_i.mean()) ** 2))
    return U, err

def block_jk(m2, m4, b):
    # block jackknife over (replica, time-block) blocks of length b
    R, S = m2.shape
    nb = S // b
    x2 = m2[:, :nb * b].reshape(R, nb, b).sum(axis=2).ravel()
    x4 = m4[:, :nb * b].reshape(R, nb, b).sum(axis=2).ravel()
    n = R * nb * b
    T2 = x2.sum(); T4 = x4.sum()
    U = 1 - (T4 / n) / (3 * (T2 / n) ** 2)
    nn = n - b
    B_i = (T2 - x2) / nn; A_i = (T4 - x4) / nn
    U_i = 1 - A_i / (3 * B_i ** 2)
    K = len(x2)
    err = math.sqrt((K - 1) / K * np.sum((U_i - U_i.mean()) ** 2))
    return U, err

def tau_int(x):
    R, S = x.shape
    y = x - x.mean()
    nfft = 1 << (2 * S - 1).bit_length()
    F = np.fft.rfft(y, n=nfft, axis=1)
    acf = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :S].mean(axis=0)
    acf = acf / acf[0]
    tau = 0.5
    for W in range(1, S):
        tau = 0.5 + acf[1:W + 1].sum()
        if W >= 5 * tau:
            break
    return float(tau)
'''
open('/work/ising_lib.py', 'w').write(lib)

import sys
sys.path.insert(0, '/work')
import importlib
import ising_lib as L
import numpy as np, json, time

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16; R = 2000; nburn = 500; nmeas = 2000
t_all = time.time()
for i, T in enumerate(Ts[:7]):
    seed = 5000 + i
    m2, m4, hist, el, itm = L.run_chain(N, R, T, nburn, nmeas, seed)
    U, e_rep = L.jack_U_rep(m2.sum(1), m4.sum(1), nmeas)
    U_b, e_blk = L.block_jk(m2, m4, 100)
    tau2 = L.tau_int(m2); tau4 = L.tau_int(m4)
    blk = [round(float(hist[a:b].mean()), 5) for a, b in [(0, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2500)]]
    res = dict(T=T, N=N, R=R, nburn=nburn, nmeas=nmeas, seed=seed, U=float(U), err_rep=float(e_rep),
               U_blockjk=float(U_b), err_blk=float(e_blk), tau_m2=tau2, tau_m4=tau4,
               m2=float(m2.mean()), m4=float(m4.mean()), iters=itm, time_s=el, hist_blocks=blk)
    json.dump(res, open(f'/work/prod_T{T:.1f}.json', 'w'), indent=1)
    print(f"T={T}: U={U:.5f} err_rep={e_rep:.5f} U_blk={U_b:.5f} err_blk={e_blk:.5f} tau2={tau2:.2f} tau4={tau4:.2f} <m2>={m2.mean():.4f} <m4>={m4.mean():.4f} iters={itm:.1f} time={el:.0f}s burn-check={blk}", flush=True)
print("total time", time.time() - t_all, flush=True)
