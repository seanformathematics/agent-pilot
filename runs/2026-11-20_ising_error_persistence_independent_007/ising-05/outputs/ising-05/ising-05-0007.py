import numpy as np, glob
files={1.9:'/work/T1.90_s13438.npz',2.0:'/work/T2.00_s20007.npz',2.1:'/work/T2.10_s21007.npz',
       2.2:'/work/T2.20_s22007.npz',2.3:'/work/T2.30_s23007.npz',2.4:'/work/T2.40_s24007.npz'}
import os
print(os.listdir('/work'))
for T,f in files.items():
    d=np.load(f)
    m2=d['m2b'].mean(axis=1); m4=d['m4b'].mean(axis=1)  # per replica
    R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas
    S2=m2.sum(); S4=m4.sum()
    Uj=np.empty(R)
    for i in range(R):
        a2=(S2-m2[i])/(R-1); a4=(S4-m4[i])/(R-1)
        Uj[i]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # also check with replica-group jackknife (20 groups)
    G=20; g=np.arange(R)%G
    Ug=[]
    for k in range(G):
        msk=g!=k
        Ug.append(1-m4[msk].mean()/(3*m2[msk].mean()**2))
    Ug=np.array(Ug)
    errg=np.sqrt((G-1)/G*np.sum((Ug-Ug.mean())**2))
    print(f"T={T} seed={int(d['seed'])} U={U:.5f} err_jk_rep={err:.5f} err_jk_20grp={errg:.5f} nsamp={d['m2b'].size}")
