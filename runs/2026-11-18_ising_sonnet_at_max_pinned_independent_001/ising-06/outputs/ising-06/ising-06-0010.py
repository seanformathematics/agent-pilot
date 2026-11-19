
import sys, json
sys.path.insert(0,'/work')
import numpy as np, run
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out=[]
allseeds=[]
for T in Ts:
    d=np.load(f"/work/prod_T{T:.1f}.npz")
    X=d['X']; nrep=X.shape[1]; nb=X.shape[2]; B=int(d['B'])
    nsamp=nrep*nb*B
    U,err=run.jack_U(X,2,3)       # cluster-improved estimators, delete-one-replica jackknife
    Up,Upe=run.jack_U(X,0,1)      # plain estimator (cross-check)
    allseeds += [int(s) for s in d['seeds']]
    out.append(dict(T=T,U=float(U),U_err=float(err),n_samples=int(nsamp),U_plain=float(Up),U_plain_err=float(Upe)))
    print(f"T={T}: U={U:.7f}  U_err={err:.7f}  n_samples={nsamp}   (plain: {Up:.7f} +- {Upe:.7f})")
json.dump(dict(points=out, seeds=allseeds), open('/work/final_results.json','w'), indent=1)
print("max U_err =", max(o['U_err'] for o in out), " min U_err =", min(o['U_err'] for o in out))
print("seeds:", allseeds)
