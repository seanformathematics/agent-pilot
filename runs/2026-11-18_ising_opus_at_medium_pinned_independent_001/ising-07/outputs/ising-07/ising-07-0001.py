
import numpy as np, time
N=16
def run(T,R=1000,neq=2000,nmeas=4000,seed=0):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),np.int8)
    ii,jj=np.indices((N,N)); masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={4:np.exp(-8/T),2:np.exp(-4/T)}
    tab=np.ones(9,np.float32); tab[8]=np.exp(-16/T)
    # dE = 2 s h, h in {-4..4}; prob = min(1,exp(-dE/T))
    pt=np.array([min(1,np.exp(-2*x/T)) for x in range(-4,5)],np.float32)  # index s*h+4
    m2=np.zeros(R);m4=np.zeros(R)
    for t in range(neq+nmeas):
        for mk in masks:
            h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x=(s*h+4)
            flip=(rng.random((R,N,N),dtype=np.float32)<pt[x])&mk
            s[flip]*=-1
        if t>=neq:
            m=s.sum((1,2),dtype=np.int32)/N**2
            m2+=m*m; m4+=m**4
    m2/=nmeas;m4/=nmeas
    M2=m2.mean();M4=m4.mean();U=1-M4/(3*M2**2)
    jk=[]
    G=50;idx=np.array_split(np.arange(R),G)
    for g in idx:
        keep=np.ones(R,bool);keep[g]=False
        jk.append(1-m4[keep].mean()/(3*m2[keep].mean()**2))
    jk=np.array(jk);err=np.sqrt((G-1)/G*((jk-jk.mean())**2).sum())
    return U,err,R*nmeas
t=time.time();print(run(2.3,R=500,neq=200,nmeas=200),time.time()-t)
