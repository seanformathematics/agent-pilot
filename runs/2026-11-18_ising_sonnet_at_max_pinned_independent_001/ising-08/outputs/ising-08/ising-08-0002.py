
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

# (1) check exact torus formula against brute force enumeration at N=4
res4, E4, M4 = exact_enum(4, Ts)
for T in (1.5, 2.3, 2.8):
    K = 1.0 / T
    Zb = np.exp(-(E4) / T).sum()      # brute force Z
    Zf = Z_torus(K, 4)
    e_formula = E_exact_torus(T, 4)
    print(f"T={T}: Z brute={Zb:.10e} formula={Zf:.10e} ratio={Zb/Zf:.12f}; E/N^2 enum={res4[T][2]:.10f} formula={e_formula:.10f}")

# (2) SW at N=4 vs exact enumeration
print("\nN=4 SW vs exact (R=256 replicas, 400 burn, 20 blocks x 100 sweeps)")
print(" T   | U_exact   | U_plain +- err     z | U_imp +- err      z | E_exact  E_sim +- err   z")
t0 = time.time()
for i, T in enumerate(Ts):
    R, nb, bl = 256, 20, 100
    out = run_sw(4, T, R, 400, nb, bl, seed=5000 + i)
    per = out.sum(axis=1)      # (q, R) per-replica sums
    n = nb * bl
    Up, ep, _ = U_from_units(per[0], per[1])
    Ui, ei, _ = U_from_units(per[2], per[3])
    Em, Ee = mean_from_units(per[4], n)
    Uex = res4[T][3]; Eex = res4[T][2]
    print(f"{T:.1f} | {Uex:.5f} | {Up:.5f} +- {ep:.5f} {(Up-Uex)/ep:+.2f} | {Ui:.5f} +- {ei:.5f} {(Ui-Uex)/ei:+.2f} | {Eex:.5f} {Em:.5f} +- {Ee:.5f} {(Em-Eex)/Ee:+.2f}")
print("time", time.time() - t0)
