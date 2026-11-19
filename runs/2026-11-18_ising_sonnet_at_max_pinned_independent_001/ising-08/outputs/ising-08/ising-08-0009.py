
import sys, time
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

N = 16
K = 30
R, burn, nb, bl = 64, 300, 10, 100
n = nb * bl
allT = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for T in (1.5, 2.0, 2.3, 2.5, 2.8):
    # production value
    prod = np.load(f'/work/prod/sums_T{T}.npy').sum(axis=1)
    Uprod, eprod, _ = U_from_units(prod[2], prod[3], 4000)
    iT = allT.index(T)
    Ui_l, ei_l, Up_l, ep_l = [], [], [], []
    for k in range(K):
        out = run_sw(N, T, R, burn, nb, bl, seed=30000 + 100 * iT + k)
        per = out.sum(axis=1)
        Ui, ei, _ = U_from_units(per[2], per[3], n)
        Up, ep, _ = U_from_units(per[0], per[1], n)
        Ui_l.append(Ui); ei_l.append(ei); Up_l.append(Up); ep_l.append(ep)
    Ui_l, ei_l, Up_l, ep_l = map(np.array, (Ui_l, ei_l, Up_l, ep_l))
    sd_i = Ui_l.std(ddof=1); sd_p = Up_l.std(ddof=1)
    zi = (Ui_l - Uprod) / np.sqrt(ei_l**2 + eprod**2)
    zp = (Up_l - Uprod) / np.sqrt(ep_l**2 + eprod**2)
    print(f"T={T}: production U_imp={Uprod:.5f}+-{eprod:.2e}")
    print(f"   improved: mean over {K} repeats={Ui_l.mean():.5f} (+-{sd_i/np.sqrt(K):.1e}), scatter sd={sd_i:.3e}, mean jackknife err={ei_l.mean():.3e}, ratio sd/err={sd_i/ei_l.mean():.2f}, rms z vs prod={np.sqrt(np.mean(zi**2)):.2f}, mean z={zi.mean():+.2f}")
    print(f"   plain   : mean over {K} repeats={Up_l.mean():.5f} (+-{sd_p/np.sqrt(K):.1e}), scatter sd={sd_p:.3e}, mean jackknife err={ep_l.mean():.3e}, ratio sd/err={sd_p/ep_l.mean():.2f}, rms z vs prod={np.sqrt(np.mean(zp**2)):.2f}, mean z={zp.mean():+.2f}  | elapsed {time.time()-t0:.0f}s", flush=True)
