import sys; sys.path.insert(0,'/work')
from wolff import run
import numpy as np
def U_and_err(M, nbin=400):
    M=M.astype(np.float64)
    n=len(M)//nbin*nbin
    x=M[:n].reshape(nbin,-1)/256.0
    m2=(x**2).mean(axis=1); m4=(x**4).mean(axis=1)
    def U(a2,a4): return 1-a4.mean()/(3*a2.mean()**2)
    Ufull=U(m2,m4)
    J=np.empty(nbin)
    S2=m2.sum(); S4=m4.sum()
    for b in range(nbin):
        J[b]=U((S2-m2[b])/(nbin-1),(S4-m4[b])/(nbin-1))
    err=np.sqrt((nbin-1)/nbin*np.sum((J-J.mean())**2))
    return Ufull, err
for idx,T in enumerate([1.9,2.0,2.1]):
    M,k=run(T,1000+idx,150,nburn=5000)
    np.save(f'/work/M_{T}.npy',M)
    U,e=U_and_err(M)
    print(T,'clusters',k,'n',len(M),'U',round(U,5),'err',round(e,5),flush=True)
