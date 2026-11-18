import sys, time, math
sys.path.insert(0, "/work")
import numpy as np, ising

plan = [(2.3, 12), (2.2, 8), (2.4, 8)]
nsteps = 250000; burn = 5000; bin_size = 1000
for T, K in plan:
    t0 = time.time()
    chainU = []; chainE = []; early = []; seeds = []
    S2all = []; S4all = []
    for c in range(K):
        seed = 30240000 + 100 * int(round(T * 10)) + c   # new seed family, distinct from production
        init = 'cold' if c % 2 == 0 else 'hot'
        out = ising.wolff_run(16, T, nsteps, seed=seed, nburn=burn, bin_size=bin_size, init=init)
        seeds.append(seed)
        Uc, ec = ising.binder_from_bins(out['S2'], out['S4'], bin_size)
        chainU.append(Uc); chainE.append(ec)
        Ue, _ = ising.binder_from_bins(out['S2'][:20], out['S4'][:20], bin_size)
        early.append(Ue)
        S2all.append(out['S2']); S4all.append(out['S4'])
    cU = np.array(chainU); cE = np.array(chainE)
    np.savez(f"/work/extra_T{T:.2f}.npz", S2=np.concatenate(S2all), S4=np.concatenate(S4all),
             chainU=cU, chainErr=cE, seeds=np.array(seeds), T=T, bin_size=bin_size)
    sd = cU.std(ddof=1)
    print(f"T={T}: K={K} chains: mean chain U={cU.mean():.5f}, sample sd={sd:.5f}, sd/sqrtK={sd/math.sqrt(K):.5f}; "
          f"mean jackknife per-chain err={cE.mean():.5f} (predicted sd of single chain); "
          f"ratio sd/err={sd/cE.mean():.2f}; early(20k) U mean={np.mean(early):.5f}; time {time.time()-t0:.1f}s", flush=True)
    print("   chain U:", np.round(cU, 5).tolist(), flush=True)
