import sys, time, math
sys.path.insert(0, "/work")
import numpy as np, ising

Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
K = 4; nsteps = 250000; burn = 5000; bin_size = 1000
for T in Tlist:
    t0 = time.time()
    S2s, S4s, SAs, seeds, chainU, chainErr = [], [], [], [], [], []
    for c in range(K):
        seed = 20240000 + 100 * int(round(T * 10)) + c
        init = 'cold' if c % 2 == 0 else 'hot'
        out = ising.wolff_run(16, T, nsteps, seed=seed, nburn=burn, bin_size=bin_size, init=init)
        seeds.append(seed)
        S2s.append(out['S2']); S4s.append(out['S4']); SAs.append(out['SA'])
        Uc, ec = ising.binder_from_bins(out['S2'], out['S4'], bin_size)
        chainU.append(Uc); chainErr.append(ec)
    S2 = np.concatenate(S2s); S4 = np.concatenate(S4s); SA = np.concatenate(SAs)
    U, err = ising.binder_from_bins(S2, S4, bin_size)
    np.savez(f"/work/prod_T{T:.2f}.npz", S2=S2, S4=S4, SA=SA, seeds=np.array(seeds),
             chainU=np.array(chainU), chainErr=np.array(chainErr), T=T, bin_size=bin_size)
    print(f"T={T}: U={U:.5f} +- {err:.5f} | chain U: {np.round(chainU,5).tolist()} chain err: {np.round(chainErr,5).tolist()} | time {time.time()-t0:.1f}s", flush=True)
