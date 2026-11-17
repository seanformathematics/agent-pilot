import sys, json, math
sys.path.insert(0, '/work')
import numpy as np
import ising

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
Ts_all = Ts
final = []
for T in Ts:
    idx = Ts_all.index(T)
    prod = np.load(f'/work/M_T{T:.2f}.npy').astype(np.int64)     # 4 chains x 5e5, concatenated
    prod_seeds = [5000000 + 100*idx + k for k in range(4)]
    prod_chains = prod.reshape(4, -1)
    if T >= 2.4:
        ext = np.load(f'/work/Mext_T{T:.2f}.npy').astype(np.int64)  # 16 chains x 5e5
        ext_seeds = [9000000 + 100*idx + k for k in range(16)]
        allM = np.concatenate([prod_chains, ext], axis=0)         # 20 x 5e5
        seeds = prod_seeds + ext_seeds
    else:
        allM = prod_chains
        seeds = prod_seeds
    nch = allM.shape[0]
    Mflat = allM.reshape(-1)
    nblk = Mflat.size // 5000
    U, Ujk, e_jk = ising.binder_jk(Mflat, nblk)
    chainU = np.array([ising.binder_from_M(c, 256) for c in allM])
    e_chain = chainU.std(ddof=1)/math.sqrt(nch)
    Uprod = ising.binder_from_M(prod, 256)
    eprod = ising.binder_jk(prod, 1000)[2]
    final.append(dict(T=T, U=float(U), U_err=float(e_jk), U_err_chain=float(e_chain), n_samples=int(Mflat.size),
                      n_chains=nch, seeds=seeds, U_production_only=float(Uprod), err_production_only=float(eprod)))
    print(f"T={T:.2f}: pooled U={U:.6f} err_jk={e_jk:.2e} err_chain={e_chain:.2e} (n_chains={nch}, n={Mflat.size}) | "
          f"production-only U={Uprod:.6f} err={eprod:.2e}", flush=True)

json.dump(final, open('/work/final_pooled.json', 'w'), indent=1)
print("max U_err:", max(f['U_err'] for f in final))
print("seeds total:", sum(len(f['seeds']) for f in final), "unique:", len(set(s for f in final for s in f['seeds'])))
