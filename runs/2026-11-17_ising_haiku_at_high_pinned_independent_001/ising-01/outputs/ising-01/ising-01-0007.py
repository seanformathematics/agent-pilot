import numpy as np
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def binder(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def jk(m2b,m4b,ng=40):
    R=m2b.shape[1]
    M2=m2b.mean(0); M4=m4b.mean(0)
    U=binder(M2,M4)
    idx=np.arange(R)%ng
    Us=np.array([binder(M2[idx!=g],M4[idx!=g]) for g in range(ng)])
    err=np.sqrt((ng-1)/ng*np.sum((Us-Us.mean())**2))
    return U,err
for T in Ts:
    d=np.load(f'/work/T{T:.2f}.npz'); U,err=jk(d['m2b'],d['m4b'])
    print(f"{T:.2f} {U:.10f} {err:.3e} nsamp={d['m2b'].shape[0]*d['m2b'].shape[1]}")
