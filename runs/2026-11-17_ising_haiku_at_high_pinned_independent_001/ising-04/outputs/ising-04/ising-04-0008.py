
import numpy as np, glob, json
B=1000
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(S2,S4,nt):
    m2=S2/nt; m4=S4/nt
    return 1-m4/(3*m2*m2)
res=[]
for T in Ts:
    fs=sorted(glob.glob(f'/work/data/T{T:.2f}_c*.npy'))
    D=[np.load(f) for f in fs]
    A=np.vstack(D); n=len(A); nt=n*B
    S2=A[:,0].sum(); S4=A[:,1].sum()
    U=U_of(S2,S4,nt)
    Uj=np.array([U_of(S2-A[i,0],S4-A[i,1],nt-B) for i in range(n)])
    err=np.sqrt((n-1)/n*np.sum((Uj-Uj.mean())**2))
    cU=[U_of(d[:,0].sum(),d[:,1].sum(),d.shape[0]*B) for d in D]
    res.append(dict(T=T,U=float(U),err=float(err),nch=len(D),chainU_std=float(np.std(cU,ddof=1)/np.sqrt(len(D))),m2=float(S2/nt),m=float(A[:,2].sum()/nt)))
    print(f"T={T:.2f} U={U:.6f} jk_err={err:.6f} chain_err={res[-1]['chainU_std']:.6f} nch={len(D)} <|m|>={res[-1]['m']:.4f}")
json.dump(res,open('/work/results.json','w'),indent=1)
