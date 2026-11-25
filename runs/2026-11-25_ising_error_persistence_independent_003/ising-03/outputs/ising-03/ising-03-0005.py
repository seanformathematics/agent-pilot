import numpy as np
R=1000
for T in [1.9,2.0,2.1]:
    d=np.load(f'/work/raw_N16_T{T}.npz'); S2=d['S2']; S4=d['S4']; cnt=int(d['cnt'])
    def U_of(mk):
        a2=S2[mk].sum(); a4=S4[mk].sum(); n=mk.sum()*cnt
        return 1-(a4/n)/(3*(a2/n)**2)
    full=np.ones(R,bool)
    jk=np.array([U_of(np.arange(R)!=r) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(T, f"{U_of(full):.6f}", f"{err:.7f}")
