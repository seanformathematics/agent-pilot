import sys
sys.path.insert(0, '/work')
import numpy as np
from scipy.special import ellipk
from sw import SW

def onsager_per_bond(T):
    K = 1.0 / T
    th = np.tanh(2 * K); coth = 1 / th
    kap = 2 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    Kell = ellipk(kap ** 2)
    return -coth * (1 + (2 / np.pi) * (2 * th ** 2 - 1) * Kell) / 2

# direct integral: g(K) = (1/(8 pi^2)) int int ln[cosh^2(2K) - sinh(2K)(cos a + cos b)] ; u_bond = -(1/2) dg/dK
def g_of_K(K, M=2048):
    th = np.arange(M) * 2 * np.pi / M
    c = np.cos(th)
    A = np.cosh(2 * K) ** 2; B = np.sinh(2 * K)
    arg = A - B * (c[:, None] + c[None, :])
    return np.mean(np.log(arg))  # equals (1/(4pi^2)) double integral; divide by 2 below
def g_norm(K, M=2048):
    return 0.5 * g_of_K(K, M) + np.log(2)   # (1/(8pi^2)) * (2pi)^2 * mean = mean/2

def u_bond_integral(T, h=1e-4, M=2048):
    K = 1.0 / T
    dg = (g_norm(K + h, M) - g_norm(K - h, M)) / (2 * h)
    return -0.5 * dg

for T in [2.0, 2.2, 2.3, 2.5, 2.8]:
    print("T=%.2f  Onsager=%.6f  integral=%.6f" % (T, onsager_per_bond(T), u_bond_integral(T)))

# 4x4 exact energy vs SW energy (algorithm check)
def exact_small_E(T, N=4):
    nconf = 1 << (N * N)
    bits = (np.arange(nconf, dtype=np.int64)[:, None] >> np.arange(N * N)) & 1
    s = (2 * bits - 1).reshape(nconf, N, N).astype(np.float64)
    E = -(s * np.roll(s, -1, axis=2)).sum(axis=(1, 2)) - (s * np.roll(s, -1, axis=1)).sum(axis=(1, 2))
    w = np.exp(-(E - E.min()) / T); w /= w.sum()
    return (w * E).sum() / (2 * N * N)
for T in [2.0, 2.5, 2.8]:
    sim = SW(4, 2000, T, seed=77)
    for _ in range(50): sim.sweep()
    es = []
    for _ in range(200):
        sim.sweep(); es.append(sim.energy().mean())
    print("4x4 T=%.2f exact E/bond=%.5f  SW E/bond=%.5f" % (T, exact_small_E(T), np.mean(es) / 32))

# naive energy function check on a random 16x16 configuration
sim = SW(16, 3, 2.5, seed=5)
sp = np.where(sim.s.reshape(3, 16, 16), 1, -1)
naive = []
for r in range(3):
    e = 0
    for i in range(16):
        for j in range(16):
            e -= sp[r, i, j] * sp[r, i, (j + 1) % 16] + sp[r, i, j] * sp[r, (i + 1) % 16, j]
    naive.append(e)
print("energy check naive:", naive, " code:", sim.energy())
