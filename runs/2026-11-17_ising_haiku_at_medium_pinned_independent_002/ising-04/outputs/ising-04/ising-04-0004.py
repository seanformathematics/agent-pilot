
import numpy as np
N=16; V=N*N
temps=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
def U_of(M):
    m=M/V
    m2=np.mean(m**2); m4=np.mean(m**4)
    return 1-m4/(3*m2**2)
for T in temps:
    M=np.load(f'/work/M_{T}.npy').astype(float)
    U=U_of(M)
    res=[]
    for nb_ in [50,100,200]:
        n=len(M)//nb_
        B=M[:n*nb_].reshape(nb_,n)
        # jackknife
        Us=np.array([U_of(np.delete(B,i,axis=0).ravel()) for i in range(nb_)])
        err=np.sqrt((nb_-1)/nb_*np.sum((Us-Us.mean())**2))
        res.append(round(err,5))
    # autocorrelation estimate of |M|
    x=np.abs(M)-np.abs(M).mean()
    ac=[np.mean(x[:-k]*x[k:])/np.var(x) for k in (1,10,50)]
    print(T, round(U,5), res, 'ac(1,10,50)=',np.round(ac,3))
