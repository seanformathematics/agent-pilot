import sys, os, time, json
sys.path.insert(0, '/work')
import numpy as np
from sw import SW

os.makedirs('/work/res', exist_ok=True)
N = 16

def run_T(T, seed, R=1024, burn=200, K=2000):
    sim = SW(N, R, T, seed=seed)
    for _ in range(burn):
        sim.sweep()
    S2 = np.zeros(R, dtype=np.int64); S4 = np.zeros(R, dtype=np.int64)
    S1 = np.zeros(R, dtype=np.int64); SE = np.zeros(R)
    prev = np.sign(sim.mag_sum()); nflip = 0
    for t in range(K):
        sim.sweep()
        M = sim.mag_sum()
        M2 = M * M
        S2 += M2; S4 += M2 * M2; S1 += np.abs(M)
        SE += sim.energy()
        sg = np.sign(M)
        nflip += int(np.count_nonzero(sg != prev)); prev = sg
    return dict(S2=S2, S4=S4, S1=S1, SE=SE, nflip=nflip, R=R, K=K)

def U_and_err(S2, S4, K):
    R = S2.size
    A2 = float(S2.sum()); A4 = float(S4.sum()); n = K * R
    U = 1 - (A4 / n) / (3 * (A2 / n) ** 2)
    B2 = A2 - S2.astype(float); B4 = A4 - S4.astype(float); nb = n - K
    Uj = 1 - (B4 / nb) / (3 * (B2 / nb) ** 2)
    var = (R - 1) / R * np.sum((Uj - Uj.mean()) ** 2)
    groups = np.array_split(np.arange(R), 8)
    Ug = []
    for g in groups:
        g2 = float(S2[g].sum()); g4 = float(S4[g].sum()); ng = K * len(g)
        Ug.append(1 - (g4 / ng) / (3 * (g2 / ng) ** 2))
    Ug = np.array(Ug)
    return U, np.sqrt(var), Ug.std(ddof=1) / np.sqrt(8)

Ts = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t_start = time.time()
for i, T in enumerate(Ts):
    t0 = time.time()
    seed = 1000 + 7 + i
    d = run_T(T, seed)
    U, Uerr, Ucheck = U_and_err(d['S2'], d['S4'], d['K'])
    n = d['R'] * d['K']
    out = dict(T=T, U=U, U_err=Uerr, U_err_groupcheck=Ucheck, n_samples=n, seed=seed,
               mean_absM_per_site=float(d['S1'].sum() / (n * 256)), mean_E_per_site=float(d['SE'].sum() / (n * 512)),
               sign_flip_frac=d['nflip'] / n)
    np.savez('/work/res/rep0_T%.2f.npz' % T, S2=d['S2'], S4=d['S4'], S1=d['S1'], SE=d['SE'], K=d['K'], R=d['R'], seed=seed)
    with open('/work/res/summary_rep0_T%.2f.json' % T, 'w') as f:
        json.dump(out, f)
    print("T=%.2f U=%.5f +- %.5f (groupcheck %.5f) n=%d |m|=%.4f e=%.4f flipfrac=%.4f time=%.1fs" % (
        T, U, Uerr, Ucheck, n, out['mean_absM_per_site'], out['mean_E_per_site'], out['sign_flip_frac'], time.time() - t0), flush=True)
print("total %.1f s" % (time.time() - t_start))
