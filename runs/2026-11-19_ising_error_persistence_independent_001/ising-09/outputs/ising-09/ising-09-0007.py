import numpy as np
R=1000; n=18000
for T in (1.9,2.0,2.1,2.2,2.3,2.4):
    d=np.load(f"/work/res_T{T}.npz"); S2=d["S2"]; S4=d["S4"]
    t2=S2.sum(); t4=S4.sum()
    U=1-(t4/(R*n))/(3*(t2/(R*n))**2)
    jk=np.array([1-((t4-S4[r])/((R-1)*n))/(3*((t2-S2[r])/((R-1)*n))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(T, f"{U:.6f}", f"{err:.6f}")