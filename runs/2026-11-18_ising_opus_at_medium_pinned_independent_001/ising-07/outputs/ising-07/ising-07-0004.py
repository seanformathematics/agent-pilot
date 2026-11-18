
import numpy as np
N=16
def run(T,R=300,neq=3000,nmeas=3000,seed=0):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],np.int8),(R,N,N))
    ii,jj=np.indices((N,N)); masks=[((ii+jj)%2==p) for p in (0,1)]
    pt=np.array([min(1,np.exp(-2*x/T)) for x in range(-4,5)],np.float32)
    m2=np.zeros(R);m4=np.zeros(R)
    for t in range(neq+nmeas):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            flip=(rng.random((R,N,N),dtype=np.float32)<pt[s*h+4])&mk
            s[flip]*=-1
        if t>=neq:
            m=s.sum((1,2),dtype=np.int32)/N**2; m2+=m*m; m4+=m**4
    m2/=nmeas;m4/=nmeas
    U=1-m4.mean()/(3*m2.mean()**2)
    jk=np.array([1-np.delete(m4,g).mean()/(3*np.delete(m2,g).mean()**2) for g in np.array_split(np.arange(R),30)])
    return U,np.sqrt(29/30*((jk-jk.mean())**2).sum())
for T,sd in [(2.0,7),(2.3,8),(2.5,9)]: print(T,run(T,seed=sd),flush=True)
