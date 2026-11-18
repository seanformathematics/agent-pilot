
import numpy as np, time, json, sys
N=16;R=128
def run(T,seed,neq=2000,nm=10000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    ii,jj=np.indices((N,N));masks=[((ii+jj)%2==k) for k in (0,1)]
    pr=np.exp(-2*np.arange(0,5)*2/T*0.5*1.0) # placeholder
    def sweep():
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            acc=rng.random(s.shape)<np.exp(-dE/T)
            s[:]=np.where(m&acc,-s,s)
    for _ in range(neq): sweep()
    m2=np.zeros(R);m4=np.zeros(R)
    for _ in range(nm):
        sweep()
        m=s.sum((1,2))/N**2
        m2+=m**2;m4+=m**4
    m2/=nm;m4/=nm
    U=1-m4.sum()/(3*m2.sum()**2/R)*1.0
    # jackknife over replicas
    Uj=[]
    for r in range(R):
        k=np.arange(R)!=r
        Uj.append(1-m4[k].mean()/(3*m2[k].mean()**2))
    Uj=np.array(Uj)
    err=np.sqrt((R-1)/R*((Uj-Uj.mean())**2).sum())
    return 1-m4.mean()/(3*m2.mean()**2),err
t=time.time()
print(run(2.3,1,200,500),time.time()-t)
