
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
res4, E4, M4 = exact_enum(4, Ts)

t0 = time.time()
print("SW N=4: R=256, burn 300, 40 blocks x 250 sweeps (2.56M samples per T), new seeds")
zs_p, zs_i = [], []
for i, T in enumerate(Ts):
    R, nb, bl = 256, 40, 250
    out = run_sw(4, T, R, 300, nb, bl, seed=7000 + i)
    per = out.sum(axis=1); n = nb * bl
    Up, ep, bp = U_from_units(per[0], per[1], n)
    Ui, ei, bi = U_from_units(per[2], per[3], n)
    Uex = res4[T][3]
    zs_p.append((Up - Uex) / ep); zs_i.append((Ui - Uex) / ei)
    print(f"{T:.1f} | exact {Uex:.5f} | plain {Up:.5f} +- {ep:.5f} z={zs_p[-1]:+.2f} | impr {Ui:.5f} +- {ei:.5f} z={zs_i[-1]:+.2f}")
print("mean z plain %.2f  improved %.2f; chi2/14 plain %.2f improved %.2f; time %.0f s" % (np.mean(zs_p), np.mean(zs_i), np.sum(np.square(zs_p))/14, np.sum(np.square(zs_i))/14, time.time()-t0))

t0 = time.time()
print("\nMetropolis N=4: R=256, burn 500, 20 blocks x 100 sweeps (512k samples per T)")
zs = []
for i, T in enumerate(Ts):
    R, nb, bl = 256, 20, 100
    out = run_metro(4, T, R, 500, nb, bl, seed=9000 + i)
    per = out.sum(axis=1); n = nb * bl
    Up, ep, bp = U_from_units(per[0], per[1], n)
    Em, Ee = mean_from_units(per[2], n)
    Uex = res4[T][3]; Eex = res4[T][2]
    zs.append((Up - Uex) / ep)
    print(f"{T:.1f} | exact {Uex:.5f} | metro {Up:.5f} +- {ep:.5f} z={zs[-1]:+.2f} | E exact {Eex:.5f} metro {Em:.5f} +- {Ee:.5f} z={(Em-Eex)/Ee:+.2f}")
print("mean z %.2f chi2/14 %.2f; time %.0f s" % (np.mean(zs), np.sum(np.square(zs))/14, time.time()-t0))
