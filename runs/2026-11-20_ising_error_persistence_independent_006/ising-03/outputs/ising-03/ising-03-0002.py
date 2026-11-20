
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000; EQ=2000; MEAS=18000
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
out={}
for k,T in enumerate(Ts):
    seed=1000+k
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    t0=time.time()
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    # per-replica time series of m, store block sums
    blocks=[]
    for sw in range(EQ+MEAS):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE=2*s*nb
            pr=np.exp(-np.clip(dE,0,None)/T)
            flip=(rng.random(s.shape)<pr)&m
            s=np.where(flip,-s,s).astype(np.int8)
        if sw>=EQ:
            mag=s.reshape(R,-1).mean(axis=1)
            M2+=mag**2; M4+=mag**4; cnt+=1
    m2=M2/cnt; m4=M4/cnt
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4,cnt=cnt,seed=seed)
    print(T, time.time()-t0, flush=True)
