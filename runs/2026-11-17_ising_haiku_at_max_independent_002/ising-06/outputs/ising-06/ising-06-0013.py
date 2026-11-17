import sys, time
sys.path.insert(0, '/work')
import numpy as np
import ising

Ts_all = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
for T in [2.4, 2.5]:
    idx = Ts_all.index(T)
    t0 = time.time()
    Ms = []
    for k in range(16):
        seed = 9000000 + 100*idx + k
        M, _ = ising.wolff_chain(16, T, 500000, 5000, seed=seed)
        Ms.append(M.astype(np.int16))
    arr = np.stack(Ms)
    np.save(f'/work/Mext_T{T:.2f}.npy', arr)
    Us = np.array([ising.binder_from_M(m.astype(np.int64), 256) for m in arr])
    print(f"T={T}: 16 extra chains done in {time.time()-t0:.1f}s; chain U mean={Us.mean():.6f}, std={Us.std(ddof=1):.6f}", flush=True)
