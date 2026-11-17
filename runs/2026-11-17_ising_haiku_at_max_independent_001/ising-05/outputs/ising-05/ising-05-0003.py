import sys, time, math
sys.path.insert(0, '/work')
src = open('/work/wolff.py').read()
src = src.replace('''def jackknife_U(S2, S4, G):
    """Delete-one-group jackknife over G groups of consecutive blocks.
    Returns (U_full, err, U_jk_mean)."""
    nb = len(S2)
    gsz = nb // G
    used = gsz * G
    S2g = [sum(S2[k * gsz:(k + 1) * gsz]) for k in range(G)]
    S4g = [sum(S4[k * gsz:(k + 1) * gsz]) for k in range(G)]
    tot2 = sum(S2g[:G]); tot4 = sum(S4g[:G])
    n = used * 1.0  # per-block count normalised below via block size (counts cancel out in ratio)
    # counts per group are identical, so use group counts = gsz (in units of blocks)
    U_full = U_from_sums(tot2, tot4, G * gsz)
    Uk = []
    for k in range(G):
        Uk.append(U_from_sums(tot2 - S2g[k], tot4 - S4g[k], (G - 1) * gsz))''',
'''def jackknife_U(S2, S4, G, block):
    """Delete-one-group jackknife over G groups of consecutive blocks.
    block = number of measurements per block. Returns (U_full, err, U_jk_mean)."""
    nb = len(S2)
    gsz = nb // G
    S2g = [sum(S2[k * gsz:(k + 1) * gsz]) for k in range(G)]
    S4g = [sum(S4[k * gsz:(k + 1) * gsz]) for k in range(G)]
    tot2 = sum(S2g); tot4 = sum(S4g)
    ng = gsz * block  # measurements per group
    U_full = U_from_sums(tot2, tot4, G * ng)
    Uk = []
    for k in range(G):
        Uk.append(U_from_sums(tot2 - S2g[k], tot4 - S4g[k], (G - 1) * ng))''')
open('/work/wolff.py', 'w').write(src)
import importlib
import wolff
importlib.reload(wolff)
from wolff import wolff_chain, jackknife_U
import numpy as np

# ---- exact 4x4 check ----
N = 4; V = N * N
idx = np.arange(2 ** V)
bits = (idx[:, None] >> np.arange(V)[None, :]) & 1
spins = (bits * 2 - 1).reshape(-1, N, N).astype(np.int64)
Mtot = spins.sum(axis=(1, 2))
E = -((spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2)) + (spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2)))
Emin = E.min()
for T in [1.5, 2.2692, 2.8]:
    w = np.exp(-(E - Emin) / T); Z = w.sum()
    m2 = (w * Mtot ** 2).sum() / Z; m4 = (w * Mtot ** 4).sum() / Z
    Uex = 1 - m4 / (3 * m2 ** 2)
    S2, S4, Mf, avgsz = wolff_chain(4, T, seed=12345, n_burn=2000, n_blocks=20000, block=100)
    Uf, err, _ = jackknife_U(S2, S4, 50, 100)
    print(f"N=4 T={T}: exact U={Uex:.5f} | Wolff U={Uf:.5f} +- {err:.5f}  pull={(Uf-Uex)/err:.2f}")

# ---- pilot on N=16 ----
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t_start = time.time()
for k, T in enumerate(Ts):
    t0 = time.time()
    S2, S4, Mf, avgsz = wolff_chain(16, T, seed=1000 + k, n_burn=5000, n_blocks=2000, block=100)
    dt = time.time() - t0
    res = []
    for G in [10, 20, 50, 100]:
        Uf, err, _ = jackknife_U(S2, S4, G, 100)
        res.append(f"G{G}:{err:.5f}")
    print(f"T={T}: U={Uf:.5f}  errs[{' '.join(res)}]  avgsz={avgsz:.1f}  time={dt:.1f}s", flush=True)
print("total pilot time", round(time.time() - t_start, 1))
