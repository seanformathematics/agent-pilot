
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import sw

# ---------- exact enumeration for N=4 ----------
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

# ---------- SW check on N=4 ----------
R = 2000; nmeas = 600; neq = 50
for T, seed in [(1.5, 101), (2.0, 102), (2.3, 103), (2.8, 104)]:
    sim = sw.SwendsenWang(N, R, T, seed, start="hot")
    for _ in range(neq):
        sim.sweep(measure=False)
    acc = {k: np.zeros(R) for k in ["m2", "m4", "m2i", "m4i", "e"]}
    for _ in range(nmeas):
        o = sim.sweep(measure=True, energy=True)
        for k in acc:
            acc[k] += o[k]
    for k in acc:
        acc[k] /= nmeas
    m2x, m4x, Ux, ex = exact(T)
    U1, e1 = sw.jackknife_U(acc["m2"], acc["m4"], 100)
    U2, e2 = sw.jackknife_U(acc["m2i"], acc["m4i"], 100)
    print(f"T={T}: exact U={Ux:.5f} m2={m2x:.5f} m4={m4x:.5f} e={ex:.5f}")
    print(f"      plain   U={U1:.5f} +- {e1:.5f} (z={(U1-Ux)/e1:+.2f})  m2={acc['m2'].mean():.5f} m4={acc['m4'].mean():.5f} e={acc['e'].mean():.5f}")
    print(f"      improv  U={U2:.5f} +- {e2:.5f} (z={(U2-Ux)/e2:+.2f})  m2={acc['m2i'].mean():.5f} m4={acc['m4i'].mean():.5f}")

# ---------- timing for N=16 ----------
for R in [500, 1000, 2000]:
    sim = sw.SwendsenWang(16, R, 2.269, 7, start="hot")
    for _ in range(5):
        sim.sweep(measure=False)
    t0 = time.time()
    k = 10
    for _ in range(k):
        sim.sweep(measure=True)
    dt = (time.time() - t0) / k
    print(f"N=16 R={R}: {dt*1000:.1f} ms/sweep, {R/dt:.0f} replica-sweeps/s")
