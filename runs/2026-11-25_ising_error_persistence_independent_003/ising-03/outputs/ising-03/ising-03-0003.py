import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(T, seed, n_eq=2000, n_meas=20000, block=1000):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    P=np.array([min(1.0,np.exp(-2*p/T)) for p in range(-4,5)])
    def sweep(s):
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            prod=(s*nb+4)
            acc=(rng.random(s.shape,dtype=np.float32)<P[prod]) & mk
            s=np.where(acc,-s,s).astype(np.int8)
        return s
    for k in range(n_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    blocks=[]
    b2=np.zeros(R); b4=np.zeros(R); bc=0
    for k in range(n_meas):
        s=sweep(s)
        m=s.sum(axis=(1,2))/(N*N)
        m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
        b2+=m2; b4+=m2*m2; bc+=1
        if bc==block:
            blocks.append(float(1-(b4.sum()/bc/R)/(3*(b2.sum()/bc/R)**2)))
            b2[:]=0;b4[:]=0;bc=0
    return S2,S4,cnt,blocks
out={}
for T,seed in [(1.9,101),(2.0,102),(2.1,103)]:
    t=time.time()
    S2,S4,cnt,blocks=run(T,seed)
    np.savez(f'/work/raw_N16_T{T}.npz',S2=S2,S4=S4,cnt=cnt)
    m2=S2.sum()/cnt/R; m4=S4.sum()/cnt/R
    U=1-m4/(3*m2*m2)
    print(T, "U=",round(U,5), "blocks:",[round(b,4) for b in blocks], "time",round(time.time()-t,1), flush=True)
