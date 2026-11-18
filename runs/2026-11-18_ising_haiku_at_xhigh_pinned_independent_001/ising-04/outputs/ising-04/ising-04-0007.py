import sys, time
sys.path.insert(0,'/work')
import numpy as np, ising
Tv = [1.5, 2.0, 2.2, 2.3, 2.4, 2.6, 2.8]
R=512
rows=[]
for idx,T in enumerate(Tv):
    seed = 5000+idx
    t0=time.time()
    out = ising.run(T, R=R, n_equil=3000, n_meas=4000, seed=seed, start='cold')
    U,err = ising.binder_jk(out['A'],out['B'],out['K'])
    d = np.load(f'/work/res/T{T:.2f}_s{1000+[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8].index(T)}.npz') if False else None
    # per-replica mean |m|, min over replicas to detect stuck states
    mabs_rep = out['Mabs']/out['K']
    print(f"T={T} cold-start U={U:.5f} err={err:.5f} min_rep<|m|>={mabs_rep.min():.3f} frac(<|m|><0.5)={np.mean(mabs_rep<0.5):.4f} time={time.time()-t0:.1f}s", flush=True)
    rows.append((T,U,err))
np.save('/work/res/validation_cold.npy', np.array(rows))
