
import numpy as np,time,json,sys
def run(T,R=2000,neq=2000,nmeas=8000,every=2,seed=0):
    rng=np.random.default_rng(seed)
    L=16;s=np.ones((R,L,L),np.int8)
    ii,jj=np.indices((L,L));masks=[((ii+jj)%2==c) for c in (0,1)]
    acc=np.exp(-2*np.arange(-4,5,2)/T)  # index by (s*h+4)//2
    m2=np.zeros(R);m4=np.zeros(R);cnt=0
    for t in range(neq+nmeas):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x=(s*h+4)//2
            flip=(rng.random(s.shape)<acc[x])&mk
            s[flip]*=-1
        if t>=neq and t%every==0:
            m=s.sum((1,2),dtype=np.int32)/256.0
            m2+=m*m;m4+=m**4;cnt+=1
    m2/=cnt;m4/=cnt
    U=1-m4.mean()/(3*m2.mean()**2)
    # jackknife over replicas
    S2=m2.sum();S4=m4.sum()
    j=1-((S4-m4)/(R-1))/(3*((S2-m2)/(R-1))**2)
    err=np.sqrt((R-1)/R*((j-j.mean())**2).sum())
    return U,err,R*cnt
t0=time.time();print(run(2.3,R=500,neq=200,nmeas=1000),time.time()-t0)
