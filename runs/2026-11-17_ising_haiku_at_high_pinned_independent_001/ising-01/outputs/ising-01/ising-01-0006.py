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
    d=np.load(f'/work/T{T:.2f}.npz'); m2b,m4b=d['m2b'],d['m4b']
    U,err=jk(m2b,m4b)
    h=m2b.shape[0]//2
    U1,_=jk(m2b[:h],m4b[:h]); U2,_=jk(m2b[h:],m4b[h:])
    # alternative: replica-level Binder values averaged (sanity)
    print(f"T={T:.2f} U={U:.5f} err={err:.5f} | halves {U1:.4f} {U2:.4f} | <|m|>={np.sqrt(m2b.mean()):.4f}")
