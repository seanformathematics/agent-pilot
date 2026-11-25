
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
def make_mask():
    i,j=np.indices((N,N))
    return [( (i+j)%2==p) for p in (0,1)]
MASKS=make_mask()
def run(T,R,nsweep,nburn,seed,meas_every=1):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),dtype=np.int8)  # cold start
    # random start half? use cold start; burn-in handles it
    pacc={d:np.exp(-d/T) for d in (4,8)}
    thr=np.zeros(9); thr[4]=np.exp(-4/T); thr[8]=np.exp(-8/T)
    m2=[];m4=[];
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    tot=nburn+nsweep
    for sw in range(tot):
        for p in (0,1):
            nb=(np.roll(s,1,1).astype(np.int8)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-2+4,2)) if False else None
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb  # in {-8..8}
            u=rng.random((R,N,N),dtype=np.float32)
            acc=(dE<=0)|(u<thr[np.clip(dE,0,8)])
            acc &= MASKS[p]
            s=np.where(acc,-s,s).astype(np.int8)
        if sw>=nburn:
            m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
            S2+=m*m; S4+=m**4; cnt+=1
    return S2/cnt, S4/cnt
t=time.time()
a,b=run(2.269,512,50,0,1)
print("time per sweep R=512:",(time.time()-t)/50)
