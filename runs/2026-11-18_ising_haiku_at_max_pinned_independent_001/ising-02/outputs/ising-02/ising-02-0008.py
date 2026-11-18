
import numpy as np
def analyze(fn, label, nblocks_t=10):
    Ms = np.load(fn).astype(np.float64)
    R, K = Ms.shape
    m = Ms/256.0; A2 = m*m; A4 = A2*A2
    a2 = A2.mean(axis=1); a4 = A4.mean(axis=1)           # per replica time averages
    U_r = 1 - a4/(3*a2**2)
    Utot = 1 - A4.mean()/(3*A2.mean()**2)
    # replica jackknife error (as in production)
    s2 = a2.sum(); s4 = a4.sum()
    s2j = s2 - a2; s4j = s4 - a4
    Uj = 1 - (s4j/(R-1))/(3*(s2j/(R-1))**2)
    e_jk = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # time-block error: split time into blocks, use all replicas per block
    Kb = K//nblocks_t
    Ub = []
    for b in range(nblocks_t):
        sl = slice(b*Kb, (b+1)*Kb)
        Ub.append(1 - A4[:, sl].mean()/(3*A2[:, sl].mean()**2))
    Ub = np.array(Ub)
    e_time = Ub.std(ddof=1)/np.sqrt(nblocks_t)
    # replica-to-replica variance of per-replica a2 vs expected from temporal variance
    var_r = a2.var(ddof=1)
    # expected variance of replica mean from within-replica time series (naive, ignores autocorr)
    within = np.mean([A2[r].var(ddof=1)/K for r in range(0, R, 10)])
    print(f"--- {label}: R={R} K={K}")
    print(f"U={Utot:.5f}  jackknife err={e_jk:.5f}  time-block err (blocks of {Kb} steps, {nblocks_t} blocks)={e_time:.5f}")
    print(f"per-replica U_r: mean={U_r.mean():.5f} sd={U_r.std(ddof=1):.5f} min={U_r.min():.4f} max={U_r.max():.4f} "
          f"skew={((U_r-U_r.mean())**3).mean()/U_r.std()**3:.2f} exkurt={((U_r-U_r.mean())**4).mean()/U_r.std()**4-3:.2f}")
    print(f"var of replica-mean m2: observed/expected(indep) ratio = {var_r/(within*K/K*1.0)/ (1.0):.3f}  (a2 sd across replicas={np.sqrt(var_r):.5f})")
    # per-block replica-averaged m2 over time: independence check
    ts = A2.mean(axis=0)
    return dict(U=Utot, e_jk=e_jk, e_time=e_time, Ub=Ub)

r1 = analyze("/work/prod_M_T2.50.npy", "production T=2.5 (seed 900070)")
r2 = analyze("/work/pilot_T2.5.npy", "pilot T=2.5 (R=1000,K=1000)")
print("time-block U values (prod):", np.round(r1['Ub'], 5))
print("time-block U values (pilot):", np.round(r2['Ub'], 5))
