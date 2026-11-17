import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import SW
from scipy.special import ellipk

def onsager_per_bond(T):
    K = 1.0 / T
    th = np.tanh(2 * K); coth = 1 / th
    kap = 2 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    return -coth * (1 + (2 / np.pi) * (2 * th ** 2 - 1) * ellipk(kap ** 2)) / 2

def metropolis_energy(T, R=256, N=16, burn=300, K=1500, seed=1):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    acc_e = []
    def half(s, mask):
        nn = np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) + np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2)
        dE = 2 * s.astype(np.int16) * nn
        u = rng.random(s.shape)
        acc = (dE <= 0) | (u < np.exp(-np.clip(dE, 0, None) / T))
        flip = acc & mask[None]
        s[flip] *= -1
        return s
    def energy(s):
        sp = s.astype(np.float64)
        return -((sp * np.roll(sp, -1, axis=1)).sum(axis=(1, 2)) + (sp * np.roll(sp, -1, axis=2)).sum(axis=(1, 2)))
    for t in range(burn + K):
        s = half(s, masks[0]); s = half(s, masks[1])
        if t >= burn:
            acc_e.append(energy(s).mean())
    acc_e = np.array(acc_e)
    return acc_e.mean() / (2 * N * N), acc_e.std() / np.sqrt(len(acc_e)) / (2 * N * N)

t0 = time.time()
for T in [2.5, 2.8]:
    em, eem = metropolis_energy(T)
    print("Metropolis 16x16 T=%.2f: E/bond=%.5f +- %.5f (naive) ; Onsager=%.5f" % (T, em, eem, onsager_per_bond(T)), flush=True)
print("metropolis time %.1f s" % (time.time() - t0), flush=True)

for T in [2.5, 2.8]:
    sim = SW(16, 256, T, seed=424242)
    for _ in range(200): sim.sweep()
    es = []
    for _ in range(1000):
        sim.sweep(); es.append(sim.energy())
    es = np.array(es) / 512.0
    per_rep = es.mean(axis=0)
    print("SW 16x16 fresh T=%.2f: E/bond=%.5f +- %.5f ; Onsager=%.5f" % (T, es.mean(), per_rep.std(ddof=1) / np.sqrt(len(per_rep)), onsager_per_bond(T)), flush=True)
print("total %.1f s" % (time.time() - t0))
