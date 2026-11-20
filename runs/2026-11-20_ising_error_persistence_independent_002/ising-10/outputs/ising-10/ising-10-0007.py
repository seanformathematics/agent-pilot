
import numpy as np
def U_of(S2,S4):
    m2=S2.mean(); m4=S4.mean()
    return 1-m4/(3*m2*m2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz')
    S2,S4=d['S2'],d['S4']
    nb,R=S2.shape
    U=U_of(S2,S4)
    # replica jackknife
    a2=S2.sum(0); a4=S4.sum(0)
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.array([1-(tot4-a4[r])/(nb*(R-1))/(3*((tot2-a2[r])/(nb*(R-1)))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # time-block jackknife (groups of 10 blocks = 5000 sweeps) as check
    g=10; ng=nb//g
    Ug=[]
    for k in range(ng):
        mask=np.ones(nb,bool); mask[k*g:(k+1)*g]=False
        Ug.append(U_of(S2[mask],S4[mask]))
    Ug=np.array(Ug)
    errt=np.sqrt((ng-1)/ng*np.sum((Ug-Ug.mean())**2))
    # halves drift check
    h1=U_of(S2[:nb//2],S4[:nb//2]); h2=U_of(S2[nb//2:],S4[nb//2:])
    print(f"T={T} U={U:.5f} err_replica_jk={err:.5f} err_timeblock_jk={errt:.5f} halves={h1:.4f},{h2:.4f} <|m|>={d['S1'].mean():.4f}")
