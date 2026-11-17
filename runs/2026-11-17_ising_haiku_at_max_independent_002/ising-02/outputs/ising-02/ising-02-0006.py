import sys, time, json
sys.path.insert(0, '/work')
import numpy as np
from scipy.special import ellipk
from sw import SW

# (a) Onsager energy per bond: u_bond = -coth(2K)[1 + (2/pi)(2tanh^2(2K)-1) K(kappa)] / 2, kappa = 2 sinh(2K)/cosh^2(2K)
def onsager_per_bond(T):
    K = 1.0 / T
    th = np.tanh(2 * K); coth = 1 / th
    kap = 2 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    Kell = ellipk(kap ** 2)   # scipy uses parameter m = k^2
    per_spin = -coth * (1 + (2 / np.pi) * (2 * th ** 2 - 1) * Kell)
    return per_spin / 2

sim_e = {1.5: None}
res_json = {}
for T in [1.5, 2.0, 2.2, 2.3, 2.5, 2.8]:
    try:
        d = json.load(open('/work/res/summary_rep0_T%.2f.json' % T))
        res_json[T] = d['mean_E_per_site']
    except Exception:
        pass
for T in sorted(res_json):
    print("T=%.2f  simulated E/bond=%.5f  Onsager E/bond=%.5f  diff=%.5f" % (T, res_json[T], onsager_per_bond(T), res_json[T] - onsager_per_bond(T)))

# (b) repeated procedure with fresh seeds
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
plan = [(2.8, [8001 + i for i in range(8)]), (2.3, [8101 + i for i in range(6)])]
summary = {}
for T, seeds in plan:
    Us = []; errs = []
    for sd in seeds:
        S2, S4, K = run_T(T, sd)
        U, e = U_jk(S2, S4, K)
        Us.append(U); errs.append(e)
    Us = np.array(Us); errs = np.array(errs)
    summary[T] = dict(U_runs=Us.tolist(), jk_err_runs=errs.tolist())
    print("T=%.2f: run U values:" % T, np.round(Us, 5))
    print("   mean U over runs=%.5f, scatter sd=%.5f (SE of mean=%.5f), median jackknife err per run=%.5f" % (
        Us.mean(), Us.std(ddof=1), Us.std(ddof=1) / np.sqrt(len(Us)), np.median(errs)), flush=True)
print("elapsed %.1f s" % (time.time() - t0))
json.dump({str(k): v for k, v in summary.items()}, open('/work/res/validation_repeats.json', 'w'))
