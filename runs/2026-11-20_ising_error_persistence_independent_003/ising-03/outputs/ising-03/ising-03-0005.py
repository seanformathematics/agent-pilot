import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
NM=8000
for T in Ts:
    d=np.load(f'/work/T{T}.npz')
    S2,S4,S2h,S4h=d['S2'],d['S4'],d['S2h'],d['S4h']
    R=len(S2)
    def U_of(a2,a4):
        return 1-a4.mean()/(3*a2.mean()**2)
    a2=S2/NM; a4=S4/NM
    U=U_of(a2,a4)
    # jackknife over replicas
    tot2=a2.sum(); tot4=a4.sum()
    jk=np.empty(R)
    for i in range(R):
        m2=(tot2-a2[i])/(R-1); m4=(tot4-a4[i])/(R-1)
        jk[i]=1-m4/(3*m2*m2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # equilibration check: first half vs second half
    Uh1=U_of(S2h/(NM/2),S4h/(NM/2))
    Uh2=U_of((S2-S2h)/(NM/2),(S4-S4h)/(NM/2))
    print(f"T={T} U={U:.5f} err={err:.5f} U_firsthalf={Uh1:.4f} U_secondhalf={Uh2:.4f} <|m|>={d['SA'].mean()/NM:.4f}")
