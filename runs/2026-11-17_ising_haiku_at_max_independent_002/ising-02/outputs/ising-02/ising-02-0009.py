import sys, time, json
sys.path.insert(0, '/work')
import numpy as np
from sw import SW

def run_T(T, seed, R=512, burn=200, K=2000):
    sim = SW(16, R, T, seed=seed)
    for _ in range(burn):
        sim.sweep()
    S2 = np.zeros(R, dtype=np.int64); S4 = np.zeros(R, dtype=np.int64)
    for t in range(K):
        sim.sweep()
        M = sim.mag_sum(); M2 = M * M
        S2 += M2; S4 += M2 * M2
    return S2, S4, K

def U_jk(S2, S4, K):
    R = S2.size
    A2 = float(S2.sum()); A4 = float(S4.sum()); n = K * R
    U = 1 - (A4 / n) / (3 * (A2 / n) ** 2)
    B2 = A2 - S2.astype(float); B4 = A4 - S4.astype(float); nb = n - K
    Uj = 1 - (B4 / nb) / (3 * (B2 / nb) ** 2)
    return U, np.sqrt((R - 1) / R * np.sum((Uj - Uj.mean()) ** 2))

t0 = time.time()
out = {}
plan = [(2.8, [8201 + i for i in range(14)]), (2.5, [8301 + i for i in range(6)])]
for T, seeds in plan:
    Us = []; es = []
    for sd in seeds:
        S2, S4, K = run_T(T, sd)
        U, e = U_jk(S2, S4, K)
        Us.append(U); es.append(e)
        if time.time() - t0 > 480:
            break
    Us = np.array(Us); es = np.array(es)
    out[str(T)] = dict(U=Us.tolist(), jk=es.tolist(), seeds=seeds[:len(Us)])
    print("T=%.2f  n_runs=%d  mean U=%.5f  scatter sd=%.5f  median jackknife err (R=512)=%.5f  mean jackknife err=%.5f" % (
        T, len(Us), Us.mean(), Us.std(ddof=1), np.median(es), es.mean()), flush=True)
    json.dump(out, open('/work/res/calib_repeats.json', 'w'))
print("elapsed %.1f s" % (time.time() - t0))
