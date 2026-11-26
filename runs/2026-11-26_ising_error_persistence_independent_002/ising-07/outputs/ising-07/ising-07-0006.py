
import numpy as np
res={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz')
    B2=d['B2']; B4=d['B4']   # (nblocks, R)
    nb,R=B2.shape
    m2=B2.mean(0); m4=B4.mean(0)  # per replica averages
    def U_of(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
    U=U_of(m2,m4)
    # replica jackknife
    Uj=np.array([U_of(np.delete(m2,i),np.delete(m4,i)) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # block-level jackknife (grouping blocks into 10 groups, across replicas too) as a check
    # stationarity: first vs second half
    h=nb//2
    U1=U_of(B2[:h].mean(0),B4[:h].mean(0)); U2=U_of(B2[h:].mean(0),B4[h:].mean(0))
    # coarser grouping: 1000-sweep-equivalent blocks (10 blocks of 100)
    g=10; ng=nb//g
    Bg2=B2[:ng*g].reshape(ng,g,R).mean(1); Bg4=B4[:ng*g].reshape(ng,g,R).mean(1)
    # jackknife over replicas using grouped data gives same estimate; compare errors via replica-level sums of blocks
    Ug=np.array([U_of(np.delete(Bg2.mean(0),i),np.delete(Bg4.mean(0),i)) for i in range(R)])
    errg=np.sqrt((R-1)/R*np.sum((Ug-Ug.mean())**2))
    print(f"T={T}: U={U:.5f} +- {err:.5f}   halves: {U1:.5f} {U2:.5f}  m2={B2.mean():.4f} seed={d['seed']}")
