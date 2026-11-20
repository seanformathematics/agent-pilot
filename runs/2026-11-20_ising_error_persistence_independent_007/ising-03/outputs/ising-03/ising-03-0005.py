import sys, time, numpy as np
sys.path.insert(0,"/work")
from wolff import run
for T, seed in [(2.2,1004),(2.3,1005),(2.4,1006)]:
    t=time.time()
    m2,m4,_ = run(16, T, 2_000_000, seed, 5000)
    B=2000
    nb=len(m2)//B
    b2=m2[:nb*B].reshape(nb,B).mean(1); b4=m4[:nb*B].reshape(nb,B).mean(1)
    np.savez(f"/work/T{T}.npz", m2=b2, m4=b4, seed=seed)
    U=1-m4.mean()/(3*m2.mean()**2)
    print(T, "U=",U, "time",time.time()-t, flush=True)
