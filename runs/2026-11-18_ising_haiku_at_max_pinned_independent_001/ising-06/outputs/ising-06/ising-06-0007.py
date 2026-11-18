import sys, glob, math
sys.path.insert(0, "/work")
import numpy as np, ising

files = sorted(glob.glob("/work/prod_T*.npz"), key=lambda f: float(f.split("prod_T")[1][:-4]))
def jk_stats(S2, S4, nper):
    nb = len(S2); n = nb * nper
    m2 = S2.sum()/n; m4 = S4.sum()/n
    U = 1 - m4/(3*m2*m2)
    A2 = (S2.sum() - S2)/(n - nper); A4 = (S4.sum() - S4)/(n - nper)
    Ub = 1 - A4/(3*A2*A2)
    err = math.sqrt((nb-1)/nb * np.sum((Ub-Ub.mean())**2))
    bias = (nb-1)*(Ub.mean()-U)
    return U, err, bias

print(f"{'T':>5} {'U':>9} {'U_err(jk)':>10} {'jk_bias':>9} {'chain_sd/sqrtK':>14} | err vs bin size (1k,2k,5k,10k,25k) in steps")
for f in files:
    d = np.load(f)
    T = float(d['T']); S2 = d['S2']; S4 = d['S4']; bs = int(d['bin_size'])
    U, err, bias = jk_stats(S2, S4, bs)
    cU = d['chainU']
    K = len(cU)
    chain_se = cU.std(ddof=1)/math.sqrt(K)
    errs = []
    for k in [1, 2, 5, 10, 25]:
        nb = len(S2)//k
        if nb < 20: errs.append(float('nan')); continue
        s2 = S2[:nb*k].reshape(nb, k).sum(1); s4 = S4[:nb*k].reshape(nb, k).sum(1)
        _, e, _ = jk_stats(s2, s4, bs*k)
        errs.append(e)
    print(f"{T:5.2f} {U:9.5f} {err:10.5f} {bias:9.1e} {chain_se:14.5f} | " + " ".join(f"{e:.5f}" for e in errs))
