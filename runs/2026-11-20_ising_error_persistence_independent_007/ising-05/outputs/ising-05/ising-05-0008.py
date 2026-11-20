import numpy as np
files={1.9:'/work/T1.90_s13438.npz',2.0:'/work/T2.00_s20007.npz',2.1:'/work/T2.10_s21007.npz',
       2.2:'/work/T2.20_s22007.npz',2.3:'/work/T2.30_s23007.npz',2.4:'/work/T2.40_s24007.npz'}
for T,f in files.items():
    d=np.load(f)
    m2=d['m2b'].mean(axis=1); m4=d['m4b'].mean(axis=1); R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    S2=m2.sum(); S4=m4.sum()
    Uj=np.array([1-((S4-m4[i])/(R-1))/(3*((S2-m2[i])/(R-1))**2) for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    print(f"{T} {U:.6f} {err:.6f}")
