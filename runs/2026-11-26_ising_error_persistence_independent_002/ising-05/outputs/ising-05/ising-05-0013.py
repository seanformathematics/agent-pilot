
import numpy as np, sys
sys.path.insert(0,'/work')
from ising import binder_jk
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/res_T{T}.npz')
    A2,A4,B2,B4=d['A2'],d['A4'],d['B2'],d['B4']
    U,err=binder_jk(A2,A4)
    # time-block jackknife on the 10 blocks (each block = 2000 sweeps, all replicas)
    nb=len(B2)
    Ub=np.array([1-B4[k].mean()/(3*B2[k].mean()**2) for k in range(nb)])
    jk=np.array([1-(np.delete(B4,k,0).mean())/(3*(np.delete(B2,k,0).mean())**2) for k in range(nb)])
    errb=np.sqrt((nb-1)/nb*np.sum((jk-jk.mean())**2))
    print(f"T={T} U={U:.8f} err_repl={err:.8f} err_block={errb:.8f} nrep={len(A2)} nsweeps=20000")
