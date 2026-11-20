
import numpy as np
N=16
def U_of(S2,S4,cnt):
    m2=S2.sum()/(cnt*len(S2)); m4=S4.sum()/(cnt*len(S4))
    return 1-m4/(3*m2*m2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f"/work/res_N{N}_T{T}.npz")
    S2,S4,cnt=d['S2'],d['S4'],int(d['cnt'])
    R=len(S2)
    U=U_of(S2,S4,cnt)
    # jackknife over blocks of replicas
    nb=50; idx=np.arange(R)%nb
    Uj=[]
    for b in range(nb):
        k=idx!=b
        Uj.append(U_of(S2[k],S4[k],cnt))
    Uj=np.array(Uj)
    err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
    # drift check: first half vs second half
    h=int(d['n_half'])
    U1=U_of(d['S2a'],d['S4a'],h)
    U2=U_of(S2-d['S2a'],S4-d['S4a'],cnt-h)
    print(f"T={T}: U={U:.5f} err={err:.5f}  halfU1={U1:.5f} halfU2={U2:.5f}  <|m|>={np.sqrt(S2.sum()/(cnt*R)):.4f}")
