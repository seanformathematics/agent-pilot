import numpy as np
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
def tau_int(x):
    x=x-x.mean(); n=len(x)
    f=np.fft.rfft(x,2*n); ac=np.fft.irfft(f*np.conj(f))[:n]; ac/=ac[0]
    s=0.5
    for t in range(1,n//10):
        s+=ac[t]
        if t>5*s: break
    return s
for T in Ts:
    d=np.load(f'/work/d1_{T}.npz'); m2=d['m2']; m4=d['m4']
    n=len(m2)
    out=[]
    for nb in [20,50,100,200]:
        L=n//nb
        a2=m2[:L*nb].reshape(nb,L).mean(1); a4=m4[:L*nb].reshape(nb,L).mean(1)
        U=U_of(m2,m4)
        Uj=np.array([1-np.delete(a4,i).mean()/(3*np.delete(a2,i).mean()**2) for i in range(nb)])
        err=np.sqrt((nb-1)/nb*np.sum((Uj-Uj.mean())**2))
        out.append((nb,round(err,5)))
    print(T, 'U=%.5f'%U_of(m2,m4), out, 'tau2=%.2f tau4=%.2f'%(tau_int(m2[:500000]),tau_int(m4[:500000])))
