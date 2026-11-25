
import numpy as np, time, json
N=16; R=2000; BURN=1500; MEAS=8000; M=N*N
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep(s):
        for m in masks:
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            p=np.exp(-np.maximum(dE,0)/T).astype(np.float32)
            acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<p)
            s[acc & m[None]] *= -1
        return s
    for _ in range(BURN): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); n=0
    for k in range(MEAS):
        s=sweep(s)
        m=s.reshape(R,-1).sum(1)/M; m2=m*m
        S2+=m2; S4+=m2*m2; n+=1
        if k==MEAS//2-1: S2a=S2.copy(); S4a=S4.copy(); na=n
    return S2,S4,n,S2a,S4a,na
def Ur(S2,S4,n): return 1-(S4/n)/(3*(S2/n)**2)
out={}
t0=time.time()
for T,seed in [(2.0,1002),(2.1,1003),(2.2,1004),(2.3,1005),(2.4,1006)]:
    S2,S4,n,S2a,S4a,na=run(T,seed)
    U_r=Ur(S2,S4,n); U=U_r.mean()
    jk=np.array([np.delete(U_r,i).mean() for i in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    Ua=Ur(S2a,S4a,na).mean(); Ub=Ur(S2-S2a,S4-S4a,n-na).mean()
    print(json.dumps(dict(T=T,seed=seed,U=U,err=err,n=n,Ufirst=Ua,Usecond=Ub,t=time.time()-t0)),flush=True)
    json.dump(dict(T=T,seed=seed,n=n,R=R,S2=S2.tolist(),S4=S4.tolist()),open(f'/work/res_T{T}.json','w'))
