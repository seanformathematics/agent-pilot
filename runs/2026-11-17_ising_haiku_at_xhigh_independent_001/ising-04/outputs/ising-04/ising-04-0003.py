import sys; sys.path.insert(0,'/work')
from ising import *
import numpy as np, time
N=16; R=500; T=2.269
seed=2024
rng=np.random.default_rng(seed)
masks=make_masks(N)
pac=pacc_table(T)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
t=time.time()
# burn in with checks of drift
for blk in range(10):
    for _ in range(200):
        s=sweep(s,pac,masks,rng)
    m=s.sum(axis=(1,2))/N**2
    print("burn block",blk, "mean|m|",np.abs(m).mean(), "m2",(m**2).mean())
nmeas=2000
M2=np.zeros((nmeas,R)); M4=np.zeros((nmeas,R))
for k in range(nmeas):
    s=sweep(s,pac,masks,rng)
    m=s.sum(axis=(1,2))/N**2
    M2[k]=m**2; M4[k]=m**4
print("time",time.time()-t)
np.save('/work/pilot_M2.npy',M2); np.save('/work/pilot_M4.npy',M4)
def U_of(m2,m4): return 1-m4.mean()/(3*m2.mean()**2)
U=U_of(M2,M4)
# jackknife over replicas
sum2=M2.sum(axis=0); sum4=M4.sum(axis=0)
S2=sum2.sum(); S4=sum4.sum()
jk=np.array([1-((S4-sum4[r])/(nmeas*(R-1)))/(3*((S2-sum2[r])/(nmeas*(R-1)))**2) for r in range(R)])
err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
print("U=",U,"jackknife err=",err)
# per-replica U spread and tau estimate
def tau_int(x):
    x=x-x.mean()
    n=len(x)
    f=np.fft.rfft(x,2*n)
    ac=np.fft.irfft(f*np.conj(f))[:n]
    ac/=ac[0]
    # sum until first negative
    tau=1.0
    for k in range(1,n):
        if ac[k]<0: break
        tau+=2*ac[k]
    return tau
# tau for mean m2 across time averaged over replicas
x=M2.mean(axis=1); print("tau_int m2 (replica avg series):",tau_int(x))
x=M4.mean(axis=1); print("tau_int m4 (replica avg series):",tau_int(x))
# per-replica block sizes check: error vs block of sweeps
for B in [1,10,50,100,200,500]:
    nb=nmeas//B
    b2=M2[:nb*B].reshape(nb,B,R).mean(axis=1); b4=M4[:nb*B].reshape(nb,B,R).mean(axis=1)
    # jackknife over replicas using block means
    s2=b2.sum(axis=0); s4=b4.sum(axis=0); T2=s2.sum(); T4=s4.sum()
    jk=np.array([1-((T4-s4[r])/(nb*(R-1)))/(3*((T2-s2[r])/(nb*(R-1)))**2) for r in range(R)])
    print("B",B,"err",np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2)))
