
import sys, time, os, glob
sys.path.insert(0, '/work')
import numpy as np
from sw import make_state, sw_step, measure

def run_blocks(T, N=16, R=1024, n_burn=200, n_blocks=12, steps=100, seed=1):
    rng = np.random.default_rng(seed)
    S = make_state(R, N, rng)
    for _ in range(n_burn):
        S = sw_step(S, T, rng, N)
    m2 = np.zeros((R, n_blocks)); m4 = np.zeros((R, n_blocks)); am = np.zeros((R, n_blocks)); en = np.zeros((R, n_blocks))
    for b in range(n_blocks):
        for k in range(steps):
            S = sw_step(S, T, rng, N)
            m, e = measure(S, N)
            q = m * m
            m2[:, b] += q; m4[:, b] += q * q; am[:, b] += np.abs(m); en[:, b] += e
    m2 /= steps; m4 /= steps; am /= steps; en /= steps
    return dict(m2=m2, m4=m4, am=am, en=en)

def analyze(m2, m4):
    # m2, m4: (R, nb) block means per replica; pooled over blocks equally weighted
    R = m2.shape[0]
    a2 = m2.mean(1); a4 = m4.mean(1)   # per-replica averages
    S2 = a2.sum(); S4 = a4.sum()
    U = 1 - (S4 / R) / (3 * (S2 / R) ** 2)
    Ul = 1 - ((S4 - a4) / (R - 1)) / (3 * ((S2 - a2) / (R - 1)) ** 2)
    err = np.sqrt((R - 1) / R * np.sum((Ul - Ul.mean()) ** 2))
    return U, err

t0 = time.time()
for i, T in enumerate([1.9, 2.0, 2.1, 2.2]):
    seed = 20250 + i
    d = run_blocks(T, seed=seed)
    np.savez(f'/work/res/T{T:.2f}_s{seed}.npz', T=T, seed=seed, **d)
    print(f"done T={T} t={time.time()-t0:.1f}s", flush=True)

print("---- analysis of all files ----")
for f in sorted(glob.glob('/work/res/*.npz')):
    z = np.load(f)
    m2, m4 = z['m2'], z['m4']
    U, err = analyze(m2, m4)
    # stationarity: first 4 blocks vs last 4 blocks
    U1, _ = analyze(m2[:, :4], m4[:, :4]); U2, _ = analyze(m2[:, -4:], m4[:, -4:])
    print(f"T={float(z['T']):.2f} seed={int(z['seed'])}  U={U:.5f} err={err:.5f}  U(first4)={U1:.4f} U(last4)={U2:.4f}  <|m|>={z['am'].mean():.4f} E={z['en'].mean():.4f}")
