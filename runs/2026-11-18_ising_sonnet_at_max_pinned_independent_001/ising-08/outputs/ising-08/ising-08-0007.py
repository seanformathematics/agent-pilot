
import sys, time, os, json
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

os.makedirs('/work/prod', exist_ok=True)

def run_sw_prod(N, T, R, n_burn, n_blocks, block, seed, nkeep=8):
    rng = np.random.Generator(np.random.PCG64(seed))
    sw = SWBatch(N, T, R, rng, 'mixed')
    n2 = float(N) ** 2
    for _ in range(n_burn):
        sw.step()
    out = np.zeros((6, n_blocks, R))
    ser = np.zeros((2, n_blocks * block, nkeep))      # per-sweep series (improved m2, plain m2) of the first nkeep replicas
    k_all = 0
    for b in range(n_blocks):
        acc = np.zeros((6, R))
        for k in range(block):
            S2, S4 = sw.step()
            m, E = sw.mag_energy()
            m2 = m * m
            acc[0] += m2; acc[1] += m2 * m2
            acc[2] += S2 / n2 ** 2
            acc[3] += (3.0 * S2 * S2 - 2.0 * S4) / n2 ** 4
            acc[4] += E; acc[5] += np.abs(m)
            ser[0, k_all] = (S2 / n2 ** 2)[:nkeep]; ser[1, k_all] = m2[:nkeep]; k_all += 1
        out[:, b, :] = acc
    return out, ser

def tau_int(x, c=6.0):
    x = np.asarray(x, float); x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]
    tau = 0.5
    for W in range(1, n):
        tau = 0.5 + acf[1:W + 1].sum()
        if W >= c * tau: break
    return tau

N = 16; R = 256; burn = 1000; nb = 40; bl = 100
Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
allT = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for T in Tlist:
    i = allT.index(T)
    seed = 20000 + i
    out, ser = run_sw_prod(N, T, R, burn, nb, bl, seed)
    np.save(f'/work/prod/sums_T{T}.npy', out)
    np.save(f'/work/prod/series_T{T}.npy', ser)
    per = out.sum(axis=1); n = nb * bl
    Ui, ei, bi = U_from_units(per[2], per[3], n)
    Up, ep, bp = U_from_units(per[0], per[1], n)
    Em, Ee = mean_from_units(per[4], n); Eex = E_exact_torus(T, N)
    # hot vs cold start groups
    Uc, ec, _ = U_from_units(per[2][:R//2], per[3][:R//2], n)
    Uh, eh, _ = U_from_units(per[2][R//2:], per[3][R//2:], n)
    # first vs second half in time
    h1 = out[:, :nb//2, :].sum(axis=1); h2 = out[:, nb//2:, :].sum(axis=1)
    U1, e1, _ = U_from_units(h1[2], h1[3], n // 2); U2, e2, _ = U_from_units(h2[2], h2[3], n // 2)
    taus_i = [tau_int(ser[0, :, j]) for j in range(ser.shape[2])]
    taus_p = [tau_int(ser[1, :, j]) for j in range(ser.shape[2])]
    print(f"T={T}: U_imp={Ui:.5f} +- {ei:.5f} (bias est {bi:+.1e}) | U_plain={Up:.5f} +- {ep:.5f} z(p-i)~{(Up-Ui)/np.hypot(ep,ei):+.2f}")
    print(f"      cold-start U={Uc:.5f}+-{ec:.5f}  hot-start U={Uh:.5f}+-{eh:.5f}  z={(Uc-Uh)/np.hypot(ec,eh):+.2f} | 1st half {U1:.5f}+-{e1:.5f} 2nd half {U2:.5f}+-{e2:.5f} z={(U1-U2)/np.hypot(e1,e2):+.2f}")
    print(f"      E={Em:.5f}+-{Ee:.5f} exact {Eex:.5f} z={(Em-Eex)/Ee:+.2f} | tau_int(m2_imp)={np.mean(taus_i):.2f} tau_int(m2_plain)={np.mean(taus_p):.2f} | elapsed {time.time()-t0:.0f}s", flush=True)
