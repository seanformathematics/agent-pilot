import sys, glob, json, math, os
sys.path.insert(0, "/work")
import numpy as np

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
bin_size = 1000

def jk_stats(S2, S4, nper):
    nb = len(S2); n = nb * nper
    m2 = S2.sum() / n; m4 = S4.sum() / n
    U = 1 - m4 / (3 * m2 * m2)
    A2 = (S2.sum() - S2) / (n - nper); A4 = (S4.sum() - S4) / (n - nper)
    Ub = 1 - A4 / (3 * A2 * A2)
    err = math.sqrt((nb - 1) / nb * np.sum((Ub - Ub.mean()) ** 2))
    bias = (nb - 1) * (Ub.mean() - U)
    return U, err, bias

final = []; all_seeds = []
for T in Ts:
    files = [f"/work/prod_T{T:.2f}.npz", f"/work/extra_T{T:.2f}.npz", f"/work/extra2_T{T:.2f}.npz"]
    files = [f for f in files if os.path.exists(f)]
    S2s, S4s, chainU, seeds = [], [], [], []
    for f in files:
        d = np.load(f)
        S2s.append(d['S2']); S4s.append(d['S4'])
        chainU += list(d['chainU']); seeds += list(map(int, d['seeds']))
    S2 = np.concatenate(S2s); S4 = np.concatenate(S4s)
    U, err, bias = jk_stats(S2, S4, bin_size)
    cU = np.array(chainU); K = len(cU)
    sd = cU.std(ddof=1)
    # per-chain jackknife error for validation
    chainErr = []
    for f in files:
        d = np.load(f)
        chainErr += list(d['chainErr'])
    cE = np.mean(chainErr)
    all_seeds += seeds
    final.append(dict(T=T, U=float(U), U_err=float(err), n_samples=int(len(S2) * bin_size), chains=K,
                      bias=float(bias), chain_sd_over_jk=float(sd / cE), chain_sd_pooled=float(sd / math.sqrt(K))))

print(f"{'T':>5} {'U':>13} {'U_err':>13} {'n_samples':>10} {'K':>3} {'bias':>10} {'sd/jk':>6} {'sd/sqrtK':>10}")
for r in final:
    print(f"{r['T']:5.2f} {r['U']:13.9f} {r['U_err']:13.9f} {r['n_samples']:10d} {r['chains']:3d} {r['bias']:10.2e} {r['chain_sd_over_jk']:6.2f} {r['chain_sd_pooled']:10.9f}")
print("n seeds:", len(all_seeds), "unique:", len(set(all_seeds)))
print("valid:", all(np.isfinite(r['U']) and r['U'] <= 2/3 and r['U_err'] > 0 for r in final))
json.dump(dict(final=final, seeds=all_seeds), open("/work/final.json", "w"), indent=1)
print(json.dumps(all_seeds))
