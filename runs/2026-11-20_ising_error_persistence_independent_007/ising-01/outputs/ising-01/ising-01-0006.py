import numpy as np
def Ufun(a2,a4): return 1-a4/(3*a2**2)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load('/work/ts_%.2f.npz'%T)
    M2=d['M2'].astype(np.float64); M4=d['M4'].astype(np.float64)
    a2=M2.mean(0); a4=M4.mean(0); R=len(a2)
    U=Ufun(a2.mean(),a4.mean())
    S2=a2.sum(); S4=a4.sum()
    Uj=np.array([Ufun((S2-a2[r])/(R-1),(S4-a4[r])/(R-1)) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print("%.2f %.10f %.3e %d"%(T,U,err,R*M2.shape[0]))
