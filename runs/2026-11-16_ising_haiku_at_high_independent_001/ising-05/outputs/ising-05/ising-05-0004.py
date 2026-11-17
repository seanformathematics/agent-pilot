import sys; sys.path.insert(0,'/work')
import ising, numpy as np, time, json
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res=[]
t0=time.time()
for k,T in enumerate(Ts):
    a1,a2,a4=ising.run(T,1000,2000,3000,1000+k)
    U,e=ising.U_jk(a2,a4)
    # drift check: compare replicas-average over halves would need time series; use replica split instead
    Ua=ising.U_from(a2[:500],a4[:500]); Ub=ising.U_from(a2[500:],a4[500:])
    res.append(dict(T=T,U=float(U),err=float(e),Ua=float(Ua),Ub=float(Ub),m=float(a1.mean())))
    np.savez(f'/work/T_{T}.npz',m1=a1,m2=a2,m4=a4)
    print(T,U,e,Ua,Ub,a1.mean(),time.time()-t0,flush=True)
json.dump(res,open('/work/results.json','w'),indent=1)
