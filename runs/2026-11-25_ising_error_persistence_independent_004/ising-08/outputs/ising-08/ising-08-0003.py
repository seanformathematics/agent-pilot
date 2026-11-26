
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000; T=1.9; seed=1901
EQ=3000; MEAS=8000; BLK=100
rng=np.random.default_rng(seed)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
        dE=2*s.astype(np.int16)*nn
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        s[acc&m]*=-1
t=time.time()
for k in range(EQ): sweep(s,T)
nb=MEAS//BLK
S2=np.zeros((nb,R)); S4=np.zeros((nb,R))
for b in range(nb):
    for k in range(BLK):
        sweep(s,T)
        m=s.reshape(R,-1).mean(axis=1)
        m2=m*m
        S2[b]+=m2; S4[b]+=m2*m2
    S2[b]/=BLK; S4[b]/=BLK
np.savez(f'/work/T{T}_N{N}.npz',S2=S2,S4=S4,T=T,seed=seed,EQ=EQ)
m2=S2.mean(axis=0); m4=S4.mean(axis=0)
U=1-(m4.mean())/(3*(m2.mean())**2)
print("time",time.time()-t)
print("U pooled",U)
# block-wise U over time blocks (equilibration check)
for b in range(0,nb,10):
    print(b, 1-S4[b:b+10].mean()/(3*S2[b:b+10].mean()**2))
