import numpy as np
R=1024; MEAS=10000
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/sums_T{T:.2f}.npz'); S2=d['S2']; S4=d['S4']
    Uf=lambda a2,a4: 1-a4/(3*a2**2)
    U=Uf(S2.sum()/(R*MEAS),S4.sum()/(R*MEAS))
    t2=S2.sum(); t4=S4.sum()
    jk=Uf((t2-S2)/((R-1)*MEAS),(t4-S4)/((R-1)*MEAS))
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(f"{T} U={U:.7f} err={err:.7f}")