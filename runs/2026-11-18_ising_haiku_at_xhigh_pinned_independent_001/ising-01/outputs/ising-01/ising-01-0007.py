
import sys, time, json, os
sys.path.insert(0,'/work')
import numpy as np
import ising_sw as M

Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out={}
t0=time.time()
for idx,T in enumerate(Ts):
    seed=20240600+idx
    m2b,m4b=M.run(T,seed=seed,R=256,burn=500,L=3000,block=10)
    U,e=M.jackknife_U(m2b.mean(0),m4b.mean(0))
    h=m2b.shape[0]//2
    U1,e1=M.jackknife_U(m2b[:h].mean(0),m4b[:h].mean(0))
    U2,e2=M.jackknife_U(m2b[h:].mean(0),m4b[h:].mean(0))
    out[str(T)]=dict(seed=seed,U=float(U),err=float(e),U_half1=float(U1),U_half2=float(U2),err_half=float(e1))
    np.savez(f'/work/prod_T{T}.npz',m2b=m2b,m4b=m4b)
    json.dump(out,open('/work/prod_results.json','w'),indent=1)
    print(f"T={T} seed={seed} U={U:.5f} err={e:.5f} halves={U1:.5f},{U2:.5f} t={time.time()-t0:.0f}s",flush=True)
    if time.time()-t0>480: 
        print("stopping early for time; continue next call"); break
