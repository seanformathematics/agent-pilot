import numpy as np, time, json, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T,seed,burn=1000,nsw=4000,nblk=40):
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    def sweep():
        for m in masks:
            h=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
            dE=2*S*h
            acc=(dE<=0)|(rng.random(S.shape)<np.exp(-np.maximum(dE,0)/T))
            S[:]=np.where(m&acc,-S,S)
    for _ in range(burn): sweep()
    L=nsw//nblk
    bm2=np.zeros((nblk,R)); bm4=np.zeros((nblk,R)); bm1=np.zeros((nblk,R))
    for b in range(nblk):
        for _ in range(L):
            sweep()
            mm=S.reshape(R,-1).mean(1)
            bm2[b]+=mm**2; bm4[b]+=mm**4; bm1[b]+=np.abs(mm)
    return bm2/L,bm4/L,bm1/L
Ts=[1.9,2.0,2.1,2.2,2.3,2.4]
for k,T in enumerate(Ts):
    t=time.time()
    m2,m4,m1=run(T,seed=2024+k)
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4,m1=m1)
    print(T, 'time',round(time.time()-t,1), flush=True)