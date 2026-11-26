import numpy as np
def U_and_err(M, nbin):
    M=M.astype(np.float64)
    n=len(M)//nbin*nbin
    x=M[:n].reshape(nbin,-1)/256.0
    m2=(x**2).mean(axis=1); m4=(x**4).mean(axis=1)
    U=lambda a2,a4: 1-a4.mean()/(3*a2.mean()**2)
    J=np.array([U((m2.sum()-m2[b])/(nbin-1),(m4.sum()-m4[b])/(nbin-1)) for b in range(nbin)])
    return U(m2,m4), np.sqrt((nbin-1)/nbin*np.sum((J-J.mean())**2)), len(M)
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    M=np.load(f'/work/M_{T}.npy')
    u,e,n=U_and_err(M,400); u2,e2,_=U_and_err(M,100)
    print(T, repr(u), repr(e), n, 'check nbin=100:', repr(e2))