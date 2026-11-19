
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16
R, nb, bl, burn = 128, 10, 100, 300
t0 = time.time()
print(" T  |  U_plain +- err   | U_impr +- err    | err ratio | E_sim +- err | E_exact  z")
for i, T in enumerate(Ts):
    out = run_sw(N, T, R, burn, nb, bl, seed=11000 + i)
    per = out.sum(axis=1); n = nb * bl
    Up, ep, bp = U_from_units(per[0], per[1], n)
    Ui, ei, bi = U_from_units(per[2], per[3], n)
    Em, Ee = mean_from_units(per[4], n)
    Eex = E_exact_torus(T, N)
    print(f"{T:.1f} | {Up:.4f} +- {ep:.4f} | {Ui:.4f} +- {ei:.4f} | {ep/ei:5.2f} | {Em:.5f} +- {Ee:.5f} | {Eex:.5f} {(Em-Eex)/Ee:+.2f}")
print("time", time.time() - t0)
