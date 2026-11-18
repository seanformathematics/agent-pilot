import sys, time, math
sys.path.insert(0, "/work")
import numpy as np, ising

plan = [1.5, 1.6, 1.7, 1.8]
K = 4; nsteps = 250000; burn = 5000; bin_size = 1000
for T in plan:
    t0 = time.time()
    chainU = []; chainE = []; seeds = []; S2all = []; S4all = []
    for c in range(K):
        seed = 40240000 + 100 * int(round(T * 10)) + c
        init = 'cold' if c % 2 == 0 else 'hot'
        out = ising.wolff_run(16, T, nsteps, seed=seed, nburn=burn, bin_size=bin_size, init=init)
        seeds.append(seed)
        Uc, ec = ising.binder_from_bins(out['S2'], out['S4'], bin_size)
        chainU.append(Uc); chainE.append(ec)
        S2all.append(out['S2']); S4all.append(out['S4'])
    cU = np.array(chainU); cE = np.array(chainE)
    np.savez(f"/work/extra2_T{T:.2f}.npz", S2=np.concatenate(S2all), S4=np.concatenate(S4all),
             chainU=cU, chainErr=cE, seeds=np.array(seeds), T=T, bin_size=bin_size)
    print(f"T={T}: new chains U={np.array2string(cU, precision=8)}  time {time.time()-t0:.1f}s", flush=True)
