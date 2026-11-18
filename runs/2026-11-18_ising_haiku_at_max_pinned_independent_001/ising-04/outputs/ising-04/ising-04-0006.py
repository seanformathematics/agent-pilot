
import sys, time, json, math
sys.path.insert(0, "/work")
import numpy as np
import ising

N = 16; NN = N*N
Ts_all = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 4; n_meas = 400000; burn = 5000
group = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def jackknife_pooled(Ms_list, NN, bpr=40):
    S2s = []; S4s = []
    for M in Ms_list:
        m = np.asarray(M, dtype=np.float64) / NN
        m2 = m*m; m4 = m2*m2
        L = len(m) // bpr
        S2s.append(m2[:L*bpr].reshape(bpr, L).sum(axis=1))
        S4s.append(m4[:L*bpr].reshape(bpr, L).sum(axis=1))
    S2 = np.concatenate(S2s); S4 = np.concatenate(S4s)
    B = len(S2); Lb = len(Ms_list[0]) // bpr
    T2 = S2.sum(); T4 = S4.sum(); ntot = B * Lb
    Ufull = 1.0 - (T4/ntot) / (3.0*(T2/ntot)**2)
    s2 = (T2 - S2) / ((B-1)*Lb); s4 = (T4 - S4) / ((B-1)*Lb)
    Uj = 1.0 - s4/(3.0*s2*s2)
    err = math.sqrt((B-1)/B * np.sum((Uj - Uj.mean())**2))
    return Ufull, err, ntot

out = {}
t_start = time.time()
for T in group:
    idx = Ts_all.index(T)
    t0 = time.time()
    seeds = [100003*(idx+1) + 7*r + 1 for r in range(R)]
    reps = []
    for sd in seeds:
        Mrec, spins = ising.simulate(N, T, burn, n_meas, seed=sd, start="hot")
        reps.append(Mrec)
    arr = np.stack(reps).astype(np.int16)
    np.save(f"/work/prod_T{T:.1f}.npy", arr)
    U, err, ntot = jackknife_pooled(reps, NN, bpr=40)
    Urep = [1.0 - np.mean((r/NN)**4)/(3*np.mean((r/NN)**2)**2) for r in reps]
    signfrac = np.mean([np.mean(r > 0) for r in reps])
    out[T] = dict(U=U, err=err, n=ntot, seeds=seeds, Urep=Urep, signfrac=signfrac,
                  absm=float(np.mean([np.mean(np.abs(r))/NN for r in reps])))
    print(f"T={T:.2f} U={U:.5f} err={err:.5f} n={ntot} Urep={np.round(Urep,5).tolist()} sd_rep={np.std(Urep,ddof=1):.5f} sign+={signfrac:.3f} <|m|>={out[T]['absm']:.4f} time={time.time()-t0:.1f}s", flush=True)
    if time.time() - t_start > 480:
        print("stopping early for time budget")
        break

with open("/work/results_group2.json", "w") as f:
    json.dump({str(k): v for k, v in out.items()}, f, indent=1, default=float)
print("saved", list(out.keys()))
