import sys, time, json, os
sys.path.insert(0, '/work')
import numpy as np
import ising

Ts_all = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
NCH = 4
NUPD = 500000
BURN = 5000
NBLK = 1000

def run_T(idx, T):
    Ms, Us, means = [], [], []
    seeds = []
    for k in range(NCH):
        seed = 5000000 + 100*idx + k
        seeds.append(seed)
        M, _ = ising.wolff_chain(16, T, NUPD, BURN, seed=seed)
        Ms.append(M)
        Us.append(ising.binder_from_M(M, 256))
        means.append(float(M.mean()))
    Mall = np.concatenate(Ms)
    np.save(f'/work/M_T{T:.2f}.npy', Mall.astype(np.int16))
    U_full, U_jk, U_err_jk = ising.binder_jk(Mall, NBLK)
    Us = np.array(Us)
    chain_err = Us.std(ddof=1)/np.sqrt(NCH)
    rec = dict(T=T, idx=idx, seeds=seeds, n_samples=int(Mall.size), U=float(U_full), U_jk_bias_corrected=float(U_jk),
               U_err_jackknife=float(U_err_jk), U_err_chain_scatter=float(chain_err),
               chain_U=[float(u) for u in Us], chain_mean_M=means)
    with open(f'/work/res_T{T:.2f}.json', 'w') as f:
        json.dump(rec, f, indent=1)
    return rec

todo = [1.5,1.6,1.7,1.8]
for T in todo:
    idx = Ts_all.index(T)
    t0 = time.time()
    rec = run_T(idx, T)
    print(f"T={T:.2f}: U={rec['U']:.5f} (jk-corr {rec['U_jk_bias_corrected']:.5f}) err_jk={rec['U_err_jackknife']:.5f} "
          f"err_chain={rec['U_err_chain_scatter']:.5f} chainU={np.round(rec['chain_U'],5).tolist()} "
          f"<M>={np.mean(rec['chain_mean_M']):+.2f} time={time.time()-t0:.1f}s", flush=True)
