
import numpy as np, time, sys
N=16; R=1024; T=1.9; SEED=1001; NSW=30000; BLK=500
i,j=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mA=((i+j)%2==0); mB=~mA
rng=np.random.default_rng(SEED)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
def sweep(s):
    for mask in (mA,mB):
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        s=np.where(acc&mask,-s,s)
    return s
nb=NSW//BLK
S2=np.zeros((nb,R)); S4=np.zeros((nb,R)); S1=np.zeros((nb,R))
t=time.time()
for b in range(nb):
    acc2=np.zeros(R); acc4=np.zeros(R); acc1=np.zeros(R)
    for k in range(BLK):
        s=sweep(s)
        m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2=m*m
        acc2+=m2; acc4+=m2*m2; acc1+=np.abs(m)
    S2[b]=acc2; S4[b]=acc4; S1[b]=acc1
np.savez(f'/work/raw_T{T}_N{N}.npz',S2=S2,S4=S4,S1=S1,BLK=BLK,R=R,NSW=NSW)
print("done",time.time()-t)
# quick look at U per block (pooled over chains)
Ub=1-(S4/BLK)/(3*(S2/BLK)**2)
print(np.round(Ub.mean(axis=1),4))
