import numpy as np, glob
out=[]
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T:.2f}_N16.npz')
    S2=d['S2']; S4=d['S4']   # blocks x replicas
    a2=S2.mean(0); a4=S4.mean(0)  # per-replica time averages
    R=a2.size
    U=1-a4.mean()/(3*a2.mean()**2)
    # jackknife over replicas
    tot2=a2.sum(); tot4=a4.sum()
    Uj=np.array([1-((tot4-a4[r])/(R-1))/(3*((tot2-a2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # equilibration check: first half vs second half of measurement
    nb=S2.shape[0]; h=nb//2
    U1=1-S4[:h].mean()/(3*S2[:h].mean()**2); U2=1-S4[h:].mean()/(3*S2[h:].mean()**2)
    # sign-symmetric check: mean |m| per replica fraction positive
    out.append((T,U,err,U1,U2))
    print(f"T={T} U={U:.5f} jackknife_err={err:.5f} firsthalf={U1:.5f} secondhalf={U2:.5f} n_rep={R} blocks={nb}")
