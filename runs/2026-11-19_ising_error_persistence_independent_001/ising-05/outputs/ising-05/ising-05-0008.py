import numpy as np, json
out={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}_N16.npz')
    M2,M4=d['M2'],d['M4']; nb,R=M2.shape
    U=1-M4.mean()/(3*M2.mean()**2)
    G=50; groups=np.array_split(np.arange(R),G)
    Ug=[]
    for g in groups:
        keep=np.setdiff1d(np.arange(R),g)
        a2=M2[:,keep].mean(); a4=M4[:,keep].mean()
        Ug.append(1-a4/(3*a2*a2))
    Ug=np.array(Ug)
    err=np.sqrt((G-1)/G*np.sum((Ug-Ug.mean())**2))
    # time-block jackknife cross-check (replica-averaged blocks)
    m2b=M2.mean(1); m4b=M4.mean(1)
    Bn=10; bl=np.array_split(np.arange(nb),Bn); Ub=[]
    for b in bl:
        keep=np.setdiff1d(np.arange(nb),b)
        Ub.append(1-m4b[keep].mean()/(3*m2b[keep].mean()**2))
    Ub=np.array(Ub); errb=np.sqrt((Bn-1)/Bn*np.sum((Ub-Ub.mean())**2))
    print(f"T={T} U={U:.5f} jack_rep_err={err:.6f} time_block_err={errb:.6f} nsamp={nb*R} ")
    out[T]=(U,err)
print(json.dumps({str(k):v for k,v in out.items()}))
