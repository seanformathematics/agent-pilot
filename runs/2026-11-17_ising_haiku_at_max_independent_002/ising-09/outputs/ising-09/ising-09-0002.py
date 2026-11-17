import numpy as np, math, time, json, os
os.makedirs('/work', exist_ok=True)

def wolff_step(s, rng, p, N, R, ar):
    bR = (s == np.roll(s, -1, axis=1)) & (rng.random((N, N, R), dtype=np.float32) < p)
    bD = (s == np.roll(s, -1, axis=0)) & (rng.random((N, N, R), dtype=np.float32) < p)
    bL = np.roll(bR, 1, axis=1)
    bU = np.roll(bD, 1, axis=0)
    i0 = rng.integers(0, N, R)
    j0 = rng.integers(0, N, R)
    C = np.zeros((N, N, R), dtype=bool)
    C[i0, j0, ar] = True
    iters = 0
    while True:
        iters += 1
        new = (C | np.roll(C & bR, 1, axis=1) | np.roll(C & bL, -1, axis=1)
                 | np.roll(C & bD, 1, axis=0) | np.roll(C & bU, -1, axis=0))
        if np.array_equal(new, C):
            break
        C = new
    s = np.where(C, -s, s).astype(np.int8)
    return s, iters

def run(N, R, T, nburn, nmeas, seed):
    rng = np.random.default_rng(seed)
    p = 1.0 - math.exp(-2.0 / T)
    ar = np.arange(R)
    s = np.where(rng.random((N, N, R)) < 0.5, 1, -1).astype(np.int8)
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
        m_ = m
        hist.append(float((m * m).mean()))
    el = time.time() - t0
    return m2.T.copy(), m4.T.copy(), np.array(hist), el, iters_tot / nmeas

def jack_U(S2, S4, n):
    R = S2.shape[0]
    tot2 = S2.sum(); tot4 = S4.sum()
    B = tot2 / (R * n); A = tot4 / (R * n)
    U = 1 - A / (3 * B * B)
    B_i = (tot2 - S2) / ((R - 1) * n); A_i = (tot4 - S4) / ((R - 1) * n)
    U_i = 1 - A_i / (3 * B_i ** 2)
    err = math.sqrt((R - 1) / R * np.sum((U_i - U_i.mean()) ** 2))
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
    return tau

def exact_N(N, T):
    nS = N * N
    idx = np.arange(2 ** nS)
    bits = (idx[:, None] >> np.arange(nS)[None, :]) & 1
    spins = (bits * 2 - 1).reshape(-1, N, N)
    E = -(spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2)) - (spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2))
    m = spins.sum(axis=(1, 2)) / nS
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    m2 = (w * m ** 2).sum() / Z; m4 = (w * m ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2), m2, m4

out = {}
# ---- Part A: validation on N=4 against exact enumeration
for T in [2.0, 2.5]:
    Ue, m2e, m4e = exact_N(4, T)
    m2, m4, hist, el, itm = run(4, 2000, T, 300, 2000, seed=1001)
    U, err = jack_U(m2.sum(1), m4.sum(1), m2.shape[1])
    print(f"N=4 T={T}: exact U={Ue:.5f} m2={m2e:.5f} m4={m4e:.5f} | MC U={U:.5f} +- {err:.5f} (pull {(U-Ue)/err:+.2f}) time {el:.1f}s", flush=True)
    out[f"N4_T{T}"] = dict(exact=float(Ue), mc=float(U), err=float(err))

# ---- Part B: timing and autocorrelation at N=16, T=Tc
T = 2.2692
t0 = time.time()
m2, m4, hist, el, itm = run(16, 2000, T, 300, 1000, seed=2002)
print(f"N=16 T={T}: R=2000, 1300 steps, time {el:.1f}s for 1000 meas, {el/1000*1e3:.1f} ms/step, mean flood iters {itm:.1f}", flush=True)
U, err = jack_U(m2.sum(1), m4.sum(1), m2.shape[1])
print(f"U={U:.5f} +- {err:.5f}", flush=True)
print("tau m2:", tau_int(m2), " tau m4:", tau_int(m4), flush=True)
print("hist (replica mean m2) burn-in first 10:", np.round(hist[:10], 4), " burn-in last 10 avg:", np.round(hist[290:300].mean(), 4), " meas mean:", np.round(hist[300:].mean(), 4), flush=True)
print("hist block means (100 steps each):", np.round([hist[i:i+100].mean() for i in range(0, 1300, 100)], 4), flush=True)
out["N16_T2.2692"] = dict(U=float(U), err=float(err), tau_m2=float(tau_int(m2)), tau_m4=float(tau_int(m4)), ms_per_step=float(el))
json.dump(out, open('/work/pilot.json', 'w'), indent=1)
