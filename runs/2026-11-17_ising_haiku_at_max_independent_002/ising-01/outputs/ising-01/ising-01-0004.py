
import numpy as np, time, sys
sys.path.insert(0, '/work')
from ising import wolff_run

def metro_run(N, T, R, nsweeps, burn, seed):
    rng = np.random.default_rng(seed)
    s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
    ii, jj = np.indices((N, N))
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    # acceptance prob for dE in {-8,-4,0,4,8} (dE = 2*s*nn, nn in {-4,-2,0,2,4})
    acc = {dE: min(1.0, np.exp(-dE / T)) for dE in range(-8, 9, 4)}
    Mt = np.empty((nsweeps - burn, R), dtype=np.int16)
    for sw in range(nsweeps):
        for mk in masks:
            nn = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)).astype(np.int8)
            dE = (2 * s * nn).astype(np.int16)
            pacc = np.where(dE <= 0, 1.0, np.exp(-np.maximum(dE, 0) / T))
            u = rng.random((R, N, N))
            flip = mk[None] & (u < pacc)
            s = np.where(flip, -s, s).astype(np.int8)
        if sw >= burn:
            Mt[sw - burn] = s.reshape(R, -1).sum(axis=1, dtype=np.int32)
    return Mt

def analyze(Mt, N=16):
    x = Mt.astype(np.float64) / (N * N)
    m2 = x ** 2; m4 = x ** 4
    R = x.shape[1]; nt = x.shape[0]
    S2 = m2.sum(axis=0); S4 = m4.sum(axis=0)
    tot2 = S2.sum(); tot4 = S4.sum(); tot = nt * R
    U = 1 - (tot4 / tot) / (3 * (tot2 / tot) ** 2)
    Ujk = np.empty(R)
    for r in range(R):
        a2 = (tot2 - S2[r]) / ((R - 1) * nt); a4 = (tot4 - S4[r]) / ((R - 1) * nt)
        Ujk[r] = 1 - a4 / (3 * a2 ** 2)
    err = np.sqrt((R - 1) / R * ((Ujk - Ujk.mean()) ** 2).sum())
    return U, err

t0 = time.time()
for T in [2.269, 2.8]:
    Mt_m = metro_run(16, T, 128, 6000, 1000, seed=4242 + int(T * 1000))
    Um, em = analyze(Mt_m)
    print(f"Metropolis N=16 T={T}: U={Um:.5f} +- {em:.5f} (R=128, sweeps=5000), time so far {time.time()-t0:.1f}s", flush=True)
    Mt_w, _ = wolff_run(16, T, 256, 3000, seed=777 + int(T * 1000))
    Uw, ew = analyze(Mt_w[200:])
    print(f"Wolff     N=16 T={T}: U={Uw:.5f} +- {ew:.5f}; diff={Um-Uw:.5f} ({(Um-Uw)/np.hypot(em,ew):.2f} sigma)", flush=True)

# burn-in convergence trace for Wolff at several T (fresh random starts)
for T in [1.5, 2.269, 2.8]:
    Mt, _ = wolff_run(16, T, 256, 400, seed=31337 + int(T * 1000))
    m2 = (Mt.astype(np.float64) / 256.) ** 2
    blocks = m2.reshape(8, 50, 256).mean(axis=(1, 2))
    print(f"burn trace T={T} m2 block means(50 steps): {np.round(blocks, 4)}")
print("total time", round(time.time() - t0, 1))
