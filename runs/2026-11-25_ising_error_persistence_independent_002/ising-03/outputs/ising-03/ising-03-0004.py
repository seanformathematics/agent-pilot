
import numpy as np, time, os
N=16
i,j=np.indices((N,N))
MA=((i+j)%2==0); MB=~MA

def run(T, R, nsweep, nburn, seed, meas_every=1):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int16)
    acc=np.array([1.0 if d<=0 else np.exp(-d/T) for d in range(-8,9)])
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    def half(s,mask):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        p=acc[dE+8]
        flip=(rng.random(s.shape)<p)&mask
        s[flip]*=-1
    for t in range(nburn+nsweep):
        half(s,MA); half(s,MB)
        if t>=nburn and (t-nburn)%meas_every==0:
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m
            S2+=m2; S4+=m2*m2; cnt+=1
    return S2,S4,cnt

def binder(S2,S4,cnt):
    R=len(S2)
    def U_of(idx):
        a=S2[idx].sum()/(cnt*len(idx)); b=S4[idx].sum()/(cnt*len(idx))
        return 1-b/(3*a*a)
    U=U_of(np.arange(R))
    jk=np.array([U_of(np.delete(np.arange(R),k)) for k in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    return U,err

t0=time.time()
S2,S4,c=run(2.269,256,20,0,1)
print('bench 20 sweeps R=256', time.time()-t0)
t0=time.time()
S2,S4,c=run(2.269,512,200,200,2)
print('bench 200 sweeps R=512 time', time.time()-t0)
print(binder(S2,S4,c))
