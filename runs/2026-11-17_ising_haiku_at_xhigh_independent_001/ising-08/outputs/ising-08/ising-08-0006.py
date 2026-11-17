import numpy as np, json
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
G=40
res={}
for T in temps:
    d=np.load(f'/work/raw_T{T:.2f}_N16.npz')
    S2,S4,S1=d['S2'],d['S4'],d['S1']
    R=len(S2)
    def U_of(idx):
        a2=S2[idx].mean(); a4=S4[idx].mean()
        return 1-a4/(3*a2*a2)
    U=U_of(np.arange(R))
    groups=np.array_split(np.random.default_rng(0).permutation(R),G)
    Ujk=[]
    for g in groups:
        mask=np.ones(R,bool); mask[g]=False
        Ujk.append(U_of(np.where(mask)[0]))
    Ujk=np.array(Ujk)
    err=np.sqrt((G-1)/G*np.sum((Ujk-Ujk.mean())**2))
    # per-replica scatter cross-check
    Ur=1-S4/(3*S2**2)
    err2=Ur.std(ddof=1)/np.sqrt(R)
    res[T]=dict(U=float(U),err_jk=float(err),err_rep=float(err2),mabs=float(S1.mean()))
    print(f"T={T:.2f} U={U:.5f} jackknife_err={err:.5f} replica_err={err2:.5f} <|m|>={S1.mean():.4f}")
json.dump(res,open('/work/analysis1.json','w'),indent=1)
