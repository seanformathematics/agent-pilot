import numpy as np
def ustats(S2,S4):
    m2=S2.mean(0); m4=S4.mean(0); R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    jk=np.empty(R)
    M2=m2.sum(); M4=m4.sum()
    for r in range(R):
        a=(M2-m2[r])/(R-1); b=(M4-m4[r])/(R-1)
        jk[r]=1-b/(3*a*a)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # time-block jackknife (within replicas pooled)
    nb=S2.shape[0]
    Ub=np.array([1-S4[b].mean()/(3*S2[b].mean()**2) for b in range(nb)])
    errb=Ub.std(ddof=1)/np.sqrt(nb)
    return U,err,errb
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/raw_T{T}.npz')
    U,e,eb=ustats(d['S2'],d['S4'])
    print(f"T={T} U={U:.5f} jackknife_err={e:.6f} timeblock_err={eb:.6f}")