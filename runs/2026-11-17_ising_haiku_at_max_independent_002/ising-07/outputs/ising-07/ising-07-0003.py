import numpy as np, time
t0 = time.time()
L = 16

def metro_run(T, R, nburn, nmeas, rng):
    S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, L, L))
    x = np.arange(L)[:, None]; y = np.arange(L)[None, :]
    black = ((x + y) % 2 == 0)[None]
    white = ~black
    acc4 = np.float32(np.exp(-4.0 / T)); acc8 = np.float32(np.exp(-8.0 / T))
    def half(S, mask):
        h = (np.roll(S, 1, axis=1) + np.roll(S, -1, axis=1) + np.roll(S, 1, axis=2) + np.roll(S, -1, axis=2))
        dE = 2 * S * h
        u = rng.random((R, L, L), dtype=np.float32)
        acc = (dE <= 0) | ((dE == 4) & (u < acc4)) | ((dE == 8) & (u < acc8))
        acc &= mask
        return np.where(acc, -S, S).astype(np.int8)
    for _ in range(nburn):
        S = half(S, black); S = half(S, white)
    s2 = np.zeros(R); s4 = np.zeros(R); cnt = 0
    series = []
    for _ in range(nmeas):
        S = half(S, black); S = half(S, white)
        m = S.reshape(R, -1).sum(1) / (L * L)
        s2 += m**2; s4 += m**4; cnt += 1
        series.append(m.astype(np.float32))
    return s2 / cnt, s4 / cnt, np.array(series)

def jk(s2, s4):
    R = len(s2)
    A = s4.sum(); B = s2.sum()
    U = 1 - (A / R) / (3 * (B / R)**2)
    Uj = np.array([1 - ((A - s4[i]) / (R - 1)) / (3 * ((B - s2[i]) / (R - 1))**2) for i in range(R)])
    return U, np.sqrt((R - 1) / R * np.sum((Uj - Uj.mean())**2))

rng = np.random.default_rng(777)
for T in [2.5, 2.8]:
    ts = time.time()
    s2, s4, ser = metro_run(T, 256, 1000, 6000, rng)
    U, Ue = jk(s2, s4)
    print(f"Metropolis T={T}: U={U:.5f} +- {Ue:.5f}  <m^2>={s2.mean():.5f}  <m^4>={s4.mean():.6f}  time {time.time()-ts:.1f}s")
    if time.time() - t0 > 250: break

# Pilot Wolff data sanity check
d = np.load('/work/pilot_L16.npz')
mrec = d['m'].astype(np.float64); Ts = d['Ts']; nc = 64
mr = mrec.reshape(mrec.shape[0], len(Ts), nc)
for g, T in enumerate(Ts):
    if T not in (2.0, 2.5, 2.8):
        continue
    x = mr[:, g, :]
    m2 = np.mean(x**2); m4 = np.mean(x**4)
    kurt = m4 / m2**2
    M = np.round(x*256).astype(int)
    print(f"Wolff pilot T={T}: <m^2>={m2:.5f} <m^4>={m4:.6f} kurt={kurt:.4f} U={1-kurt/3:.5f} "
          f"<|m|>={np.mean(np.abs(x)):.4f} mean m={x.mean():+.5f} parity-even M: {np.all(M%2==0)}")
print("total time", time.time()-t0)
