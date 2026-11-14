
import numpy as np, time
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
N=16;R=256
def run(T,sweeps,seed,burn=2000):
    rng=np.random.default_rng(seed)
    if T<2.3: s=np.ones((R,N,N),np.int8)*rng.choice(np.array([-1,1],np.int8),size=(R,1,1))
    else: s=rng.choice(np.array([-1,1],np.int8),size=(R,N,N))
    ii,jj=np.indices((N,N)); masks=[((ii+jj)%2==p) for p in (0,1)]
    pr={k:np.exp(-2*k/T) for k in (-4,-2,0,2,4)}
    lut=np.array([np.exp(-2*k/T) for k in range(0,9,2)])  # index by h*s/2 for h*s in 0..8? use dE=2*s*nb
    m2=np.zeros(R);m4=np.zeros(R);n=0
    for t in range(burn+sweeps):
        for mk in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            x=s*nb  # -4..4
            acc=(x<=0)|(rng.random(s.shape)<np.exp(-2*x/T))
            s=np.where(acc&mk,-s,s)
        if t>=burn:
            m=s.sum((1,2))/N**2; m2+=m**2; m4+=m**4; n+=1
    return m2/n,m4/n
def U_jk(a,b):
    M2=a.mean();M4=b.mean();U=1-M4/(3*M2**2)
    jk=[]
    for i in range(R):
        ma=(a.sum()-a[i])/(R-1);mb=(b.sum()-b[i])/(R-1);jk.append(1-mb/(3*ma**2))
    jk=np.array(jk);return U,np.sqrt((R-1)/R*((jk-jk.mean())**2).sum())
t0=time.time()
a,b=run(2.3,500,1,200);print(time.time()-t0,U_jk(a,b))
