import numpy as np, json
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
G=40
final={}
for T in temps:
    a=np.load(f'/work/raw_T{T:.2f}_N16.npz'); b=np.load(f'/work/val_T{T:.2f}_N16.npz')
    S2=np.concatenate([a['S2'],b['S2']]); S4=np.concatenate([a['S4'],b['S4']])
    R=len(S2)
    def U_of(idx):
        a2=S2[idx].mean(); a4=S4[idx].mean()
        return 1-a4/(3*a2*a2)
    U=U_of(np.arange(R))
    groups=np.array_split(np.random.default_rng(42).permutation(R),G)
    Ujk=np.array([U_of(np.setdiff1d(np.arange(R),g)) for g in groups])
    err=np.sqrt((G-1)/G*np.sum((Ujk-Ujk.mean())**2))
    Ua=1-a['S4'].mean()/(3*a['S2'].mean()**2); Ub=1-b['S4'].mean()/(3*b['S2'].mean()**2)
    final[T]=dict(U=float(U),err=float(err),U_run1=float(Ua),U_run2=float(Ub))
    print(f"T={T:.2f} U={U:.5f} err={err:.5f}  run1={Ua:.5f} run2={Ub:.5f} diff={Ua-Ub:+.5f}")
json.dump(final,open('/work/final.json','w'),indent=1)
