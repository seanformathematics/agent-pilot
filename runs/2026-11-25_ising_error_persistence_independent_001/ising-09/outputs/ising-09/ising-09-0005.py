
import numpy as np
R=512
out={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    d=np.load(f'/work/T{T}.npz')
    m2,m4=d['m2'],d['m4']
    def U(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
    u=U(m2,m4)
    # jackknife over replicas
    jk=np.empty(R)
    S2,S4=m2.sum(),m4.sum()
    for r in range(R):
        a2=(S2-m2[r])/(R-1); a4=(S4-m4[r])/(R-1)
        jk[r]=1-a4/(3*a2**2)
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    # block-check: split replicas into 8 groups of 64
    g=[U(m2[i::8],m4[i::8]) for i in range(8)]
    print(f"T={T} U={u:.5f} jackknife_err={err:.5f} groups_std={np.std(g,ddof=1)/np.sqrt(8):.5f} <m2>={m2.mean():.4f}")
