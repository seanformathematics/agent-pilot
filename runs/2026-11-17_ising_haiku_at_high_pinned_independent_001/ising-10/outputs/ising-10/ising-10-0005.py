
import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(m2,m4):
    return 1-m4/(3*m2**2)
for T in Ts:
    d=np.load(f'/work/res_{T}.npz')
    m2r=d['m2b'].mean(axis=1); m4r=d['m4b'].mean(axis=1)  # per replica
    R=len(m2r)
    U=U_of(m2r.mean(),m4r.mean())
    # jackknife over replicas
    S2=m2r.sum(); S4=m4r.sum()
    jk=np.empty(R)
    for r in range(R):
        a2=(S2-m2r[r])/(R-1); a4=(S4-m4r[r])/(R-1)
        jk[r]=U_of(a2,a4)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # also error from 8 groups of 64 replicas
    g=[]
    for k in range(8):
        sl=slice(k*64,(k+1)*64)
        g.append(U_of(m2r[sl].mean(),m4r[sl].mean()))
    g=np.array(g); err8=g.std(ddof=1)/np.sqrt(8)
    print(f"T={T} U={U:.5f} jkErr={err:.5f} groupErr={err8:.5f}")
