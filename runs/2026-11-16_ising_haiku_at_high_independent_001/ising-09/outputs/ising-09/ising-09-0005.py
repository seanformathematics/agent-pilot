
import numpy as np
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
out=[]
for T in temps:
    d=np.load(f'/work/run_T{T}.npz')
    M2=d['M2']; M4=d['M4']  # blocks x replicas
    a2=M2.mean(0); a4=M4.mean(0)  # per replica
    R=len(a2)
    S2=a2.sum(); S4=a4.sum()
    U=1-(S4/R)/(3*(S2/R)**2)
    # jackknife over replicas
    Uj=np.empty(R)
    for r in range(R):
        m2=(S2-a2[r])/(R-1); m4=(S4-a4[r])/(R-1)
        Uj[r]=1-m4/(3*m2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # block check: spread of block-level U (each block pooled over replicas)
    bl=[1-M4[b].mean()/(3*M2[b].mean()**2) for b in range(M2.shape[0])]
    out.append((T,U,err,np.std(bl,ddof=1)/np.sqrt(len(bl))))
    print(f"T={T:.1f} U={U:.6f} err={err:.6f} blockSEM={out[-1][3]:.6f} R={R}")
