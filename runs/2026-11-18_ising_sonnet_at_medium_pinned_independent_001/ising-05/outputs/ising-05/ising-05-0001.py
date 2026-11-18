
import numpy as np, time, json
N=16;R=256
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
mask=((np.indices((N,N)).sum(0))%2==0)
masks=[mask[None],~mask[None]]
def run(T,seed,neq=1000,nm=6000):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # cold-ish start half
    s[:R//2]=1
    m2=np.zeros(R);m4=np.zeros(R)
    ex=np.exp(-2*np.arange(0,9)*1.0/T)  # index by dE/2... dE=2*s*nb, nb in -4..4
    for it in range(neq+nm):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb  # -4..4
            p=np.exp(-2.0*x/T)
            acc=(rng.random(s.shape)<p)&mk
            s=np.where(acc,-s,s)
        if it>=neq:
            m=s.sum((1,2))/(N*N)
            m2+=m**2;m4+=m**4
    m2/=nm;m4/=nm
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas
    Uj=[]
    for i in range(R):
        a=np.delete(m4,i).mean();b=np.delete(m2,i).mean()
        Uj.append(1-a/(3*b*b))
    Uj=np.array(Uj)
    err=np.sqrt((R-1)/R*((Uj-Uj.mean())**2).sum())
    return U,err
t=time.time()
print(run(2.3,1,200,500),time.time()-t)
