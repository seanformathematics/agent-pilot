
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising import run, make_nb
nb = make_nb(16)
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
def U_of(m2,m4):
    return 1 - np.mean(m4)/(3*np.mean(m2)**2)
def jk(m2,m4,nb_blocks=50):
    m2=np.asarray(m2); m4=np.asarray(m4)
    n=len(m2); L=n//nb_blocks
    m2=m2[:L*nb_blocks]; m4=m4[:L*nb_blocks]
    A=m2.reshape(nb_blocks,L).mean(1); B=m4.reshape(nb_blocks,L).mean(1)
    Ut = 1 - B.sum()/ (3*(A.sum()**2)/nb_blocks/ nb_blocks * 0 + 1)  # placeholder
    tot2=A.sum()/nb_blocks; tot4=B.sum()/nb_blocks
    Ufull=1-tot4/(3*tot2**2)
    Uj=[]
    for k in range(nb_blocks):
        a=(A.sum()-A[k])/(nb_blocks-1); b=(B.sum()-B[k])/(nb_blocks-1)
        Uj.append(1-b/(3*a*a))
    Uj=np.array(Uj)
    err=np.sqrt((nb_blocks-1)/nb_blocks*np.sum((Uj-Uj.mean())**2))
    return Ufull, err
def tau_int(x, maxlag=2000):
    x=np.asarray(x)-np.mean(x); n=len(x)
    f=np.fft.rfft(x, 2*n); ac=np.fft.irfft(f*np.conj(f))[:n]; ac/=ac[0]
    t=1.0
    for k in range(1,min(maxlag,n//10)):
        if ac[k]<0.02: break
        t+=2*ac[k]
    return t
for T in Ts:
    t0=time.time()
    m2,m4=run(16,T,400000,seed=int(T*1000)+7,nburn=5000,nb=nb)
    U,e=jk(m2,m4)
    print(T, round(U,5), round(e,5), 'tau(m2)=',round(tau_int(m2),2), 'time',round(time.time()-t0,1), flush=True)
