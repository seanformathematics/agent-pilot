import sys, glob, json, math, os
sys.path.insert(0, "/work")
import numpy as np, ising

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

final = []
all_seeds = []
for T in Ts:
    dp = np.load(f"/work/prod_T{T:.2f}.npz")
    S2p, S4p = dp['S2'], dp['S4']
    seeds = list(map(int, dp['seeds'])); chains = len(dp['chainU'])
    S2s = [S2p]; S4s = [S4p]
    fe = f"/work/extra_T{T:.2f}.npz"
    if os.path.exists(fe):
        de = np.load(fe)
        S2s.append(de['S2']); S4s.append(de['S4'])
        seeds += list(map(int, de['seeds'])); chains += len(de['chainU'])
    S2 = np.concatenate(S2s); S4 = np.concatenate(S4s)
    U, err, bias = jk_stats(S2, S4, bin_size)
    Up, errp, _ = jk_stats(S2p, S4p, bin_size)
    n_meas = len(S2) * bin_size
    all_seeds += seeds
    final.append(dict(T=T, U=float(U), U_err=float(err), n_samples=int(n_meas), chains=chains,
                      bias=float(bias), U_prod=float(Up), U_err_prod=float(errp)))

print(f"{'T':>5} {'U':>11} {'U_err':>11} {'n_samples':>10} {'chains':>6} {'bias':>9} | production-only U (err)")
for r in final:
    print(f"{r['T']:5.2f} {r['U']:11.7f} {r['U_err']:11.7f} {r['n_samples']:10d} {r['chains']:6d} {r['bias']:9.1e} | {r['U_prod']:.7f} ({r['U_err_prod']:.7f})")
print("all seeds count:", len(all_seeds))
print("all finite and <= 2/3:", all(np.isfinite(r['U']) and r['U'] <= 2/3 + 1e-12 and r['U_err'] > 0 for r in final))
json.dump(dict(final=final, seeds=all_seeds), open("/work/final.json", "w"), indent=1)
