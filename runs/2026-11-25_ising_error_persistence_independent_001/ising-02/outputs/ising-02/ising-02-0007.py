
import numpy as np
def U_of(m2,m4): return 1-np.mean(m4)/(3*np.mean(m2)**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/res_T{T}_N16.npz")
    S2,S4,M,R=d['S2'],d['S4'],int(d['M']),int(d['R'])
    m2=S2.sum(0)/M; m4=S4.sum(0)/M
    U=U_of(m2,m4)
    G=40
    groups=np.array_split(np.arange(R),G)
    Uj=[]
    for g in groups:
        keep=np.ones(R,bool); keep[g]=False
        Uj.append(U_of(m2[keep],m4[keep]))
    Uj=np.array(Uj)
    err=np.sqrt((G-1)/G*np.sum((Uj-Uj.mean())**2))
    h=[U_of(S2[i]/(M/2),S4[i]/(M/2)) for i in (0,1)]
    print(f"T={T} U={U:.6f} err={err:.6f} halves={h[0]:.5f},{h[1]:.5f} <m2>={m2.mean():.4f}")
