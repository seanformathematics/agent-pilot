
import numpy as np
files={1.9:'/work/T1.9_s1901.npz',2.0:'/work/T2.0_s2001.npz',2.1:'/work/T2.1_s2101.npz',2.2:'/work/T2.2_s2201.npz',2.3:'/work/T2.3_s2301.npz',2.4:'/work/T2.4_s2401.npz'}
for T,f in files.items():
    d=np.load(f)
    S2,S4,cnt=d['S2'],d['S4'],d['cnt']
    R=S2.shape[1]
    def U_of(blocks, reps):
        a2=S2[blocks][:,reps].sum(); a4=S4[blocks][:,reps].sum()
        n=cnt[blocks].sum()*len(reps)
        m2=a2/n; m4=a4/n
        return 1-m4/(3*m2*m2)
    reps=np.arange(R)
    Ub=[U_of([b],reps) for b in range(S2.shape[0])]
    blocks=list(range(2,S2.shape[0]))
    U0=U_of(blocks,reps)
    # jackknife over replicas
    tot2=S2[blocks].sum(0); tot4=S4[blocks].sum(0); n=cnt[blocks].sum()
    A2=tot2.sum(); A4=tot4.sum()
    Uj=[]
    for r in range(R):
        a2=A2-tot2[r]; a4=A4-tot4[r]; nn=n*(R-1)
        m2=a2/nn; m4=a4/nn
        Uj.append(1-m4/(3*m2*m2))
    Uj=np.array(Uj)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(f"T={T}: U={U0:.6f} err={err:.6f}  block-U range [{min(Ub):.5f},{max(Ub):.5f}] first-block U={Ub[0]:.5f}, last={Ub[-1]:.5f}")
