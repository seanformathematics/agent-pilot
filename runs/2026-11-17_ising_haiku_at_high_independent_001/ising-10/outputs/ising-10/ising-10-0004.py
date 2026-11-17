
import numpy as np, time, os
N=16; R=512; BURN=2000; MEAS=20000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T,rng):
    for m in masks:
        nn=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)).astype(np.int8)
        dE=2*S*nn
        p=np.exp(-np.maximum(dE,0)/T)
        acc=(dE<=0)|(rng.random(S.shape)<p)
        S[...]=np.where(acc&m,-S,S)
    return S
for T in [2.6,2.7,2.8]:
    seed=1000+int(round(T*100))
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    t=time.time()
    for k in range(BURN): sweep(S,T,rng)
    s2=np.zeros(R); s4=np.zeros(R); s1=np.zeros(R)
    for k in range(MEAS):
        sweep(S,T,rng)
        m=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        s2+=m2; s4+=m2*m2; s1+=np.abs(m)
    np.savez(f'/work/T{T:.2f}.npz',s2=s2,s4=s4,s1=s1,seed=seed,T=T,MEAS=MEAS,BURN=BURN,R=R)
    M2=s2.sum()/(R*MEAS); M4=s4.sum()/(R*MEAS)
    print(T, 1-M4/(3*M2*M2), time.time()-t, flush=True)
