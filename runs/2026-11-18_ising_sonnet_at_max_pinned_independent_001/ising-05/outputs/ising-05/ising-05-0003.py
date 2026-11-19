
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import sw

N = 4; n = N * N
configs = np.arange(2 ** n)
bits = (((configs[:, None] >> np.arange(n)) & 1).astype(np.int8) * 2 - 1)
spins = bits.reshape(-1, N, N)
E = -((spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2)) + (spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2)))
Mg = bits.sum(axis=1) / n

def exact(T):
    w = np.exp(-(E - E.min()) / T); Z = w.sum()
    m2 = (w * Mg ** 2).sum() / Z; m4 = (w * Mg ** 4).sum() / Z
    e = (w * E).sum() / Z / n
    return m2, m4, 1 - m4 / (3 * m2 ** 2), e

def jk_mean(x, G=100):
    R = len(x); k = R // G
    xg = x.reshape(G, k).sum(axis=1)
    X = x.sum()
    xj = (X - xg) / (R - k)
    return x.mean(), np.sqrt((G - 1) / G * np.sum((xj - xj.mean()) ** 2))

R = 2000; nmeas = 3000; neq = 100
t0 = time.time()
for T in [1.5, 2.3, 2.8]:
    m2x, m4x, Ux, ex = exact(T)
    for seed in [201, 202]:
        sim = sw.SwendsenWang(N, R, T, seed * 10 + int(T * 10), start="hot")
        for _ in range(neq):
            sim.sweep(measure=False)
        acc = {k: np.zeros(R) for k in ["m2", "m4", "m2i", "m4i", "e"]}
        for _ in range(nmeas):
            o = sim.sweep(measure=True, energy=True)
            for k in acc:
                acc[k] += o[k]
        for k in acc:
            acc[k] /= nmeas
        res = []
        for name, key, ex_val in [("e", "e", ex), ("m2", "m2", m2x), ("m4", "m4", m4x), ("m2i", "m2i", m2x), ("m4i", "m4i", m4x)]:
            mu, er = jk_mean(acc[key])
            res.append(f"{name}: z={(mu-ex_val)/er:+.2f}")
        U1, e1 = sw.jackknife_U(acc["m2"], acc["m4"], 100)
        U2, e2 = sw.jackknife_U(acc["m2i"], acc["m4i"], 100)
        res.append(f"U plain z={(U1-Ux)/e1:+.2f} (err {e1:.5f}), U impr z={(U2-Ux)/e2:+.2f} (err {e2:.5f})")
        print(f"T={T} seed={seed}: " + "; ".join(res))
print("time", time.time() - t0)
