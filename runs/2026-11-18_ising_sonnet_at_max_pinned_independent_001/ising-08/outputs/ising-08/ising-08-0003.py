
import sys, time, re
sys.path.insert(0, '/work')
src = open('/work/ising_lib.py').read()
old = src[src.index("def U_from_units"):src.index("def mean_from_units")]
new = '''def U_from_units(u2, u4, count):
    """u2,u4: arrays (nunits,) of per-unit SUMS of m^2, m^4 over `count` measurements each.
    Returns U = 1 - <m4>/(3<m2>^2), delete-one jackknife error, jackknife bias estimate."""
    u2 = np.asarray(u2, float) / count; u4 = np.asarray(u4, float) / count
    n = u2.size
    t2 = u2.sum(); t4 = u4.sum()
    U = 1.0 - (t4 / n) / (3.0 * (t2 / n) ** 2)
    l2 = (t2 - u2) / (n - 1); l4 = (t4 - u4) / (n - 1)
    Uj = 1.0 - l4 / (3.0 * l2 ** 2)
    err = np.sqrt((n - 1) / n * np.sum((Uj - Uj.mean()) ** 2))
    bias = (n - 1) * (Uj.mean() - U)
    return U, err, bias


'''
src = src.replace(old, new)
open('/work/ising_lib.py', 'w').write(src)

import numpy as np
import importlib, ising_lib
importlib.reload(ising_lib)
from ising_lib import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
res4, E4, M4 = exact_enum(4, Ts)
print("N=4 SW vs exact (R=256 replicas, 400 burn, 20 blocks x 100 sweeps)")
print(" T   | U_exact   | U_plain +- err     z | U_imp +- err      z")
zs_p, zs_i = [], []
for i, T in enumerate(Ts):
    R, nb, bl = 256, 20, 100
    out = run_sw(4, T, R, 400, nb, bl, seed=5000 + i)
    per = out.sum(axis=1)
    n = nb * bl
    Up, ep, bp = U_from_units(per[0], per[1], n)
    Ui, ei, bi = U_from_units(per[2], per[3], n)
    Uex = res4[T][3]
    zs_p.append((Up - Uex) / ep); zs_i.append((Ui - Uex) / ei)
    print(f"{T:.1f} | {Uex:.5f} | {Up:.5f} +- {ep:.5f} {zs_p[-1]:+.2f} | {Ui:.5f} +- {ei:.5f} {zs_i[-1]:+.2f}")
print("chi2/14 plain:", np.sum(np.square(zs_p)) / 14, " improved:", np.sum(np.square(zs_i)) / 14)
