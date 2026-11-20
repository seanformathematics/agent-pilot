import numpy as np, time, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def run(T):
    SEED=1000+int(round(T*100))
    rng=np.random.default_rng(SEED)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    def sweep(s):
        for m in masks:
            nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            dE = (2*s*nn).astype(np.int16)
            pacc = np.exp(-np.maximum(dE,0)/T).astype(np.float32)
            acc = (dE<=0) | (rng.random(s.shape,dtype=np.float32) < pacc)
            s[acc & m] *= -1
        return s
    TB=4000; TM=16000; BLK=500
    for k in range(TB): sweep(s)
    nb=TM//BLK
    S2=np.zeros((nb,R)); S4=np.zeros((nb,R))
    for b in range(nb):
        for k in range(BLK):
            sweep(s)
            m=s.reshape(R,-1).mean(axis=1)
            S2[b]+=m*m; S4[b]+=m**4
        S2[b]/=BLK; S4[b]/=BLK
    np.savez(f'/work/T{T:.2f}_N16.npz',S2=S2,S4=S4,T=T,seed=SEED,TB=TB,TM=TM,BLK=BLK)
    m2=S2.mean(0); m4=S4.mean(0)
    print(T,"U=",1-m4.mean()/(3*m2.mean()**2),"blockU:",np.round([1-S4[b].mean()/(3*S2[b].mean()**2) for b in range(nb)],4))
t=time.time()
for T in [2.3,2.4]:
    run(T); print("elapsed",time.time()-t)
