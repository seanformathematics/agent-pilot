
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16; R=1000; T=1.9; seed=101
NEQ=5000; NSW=60000; BL=500
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
pt=np.exp(-2*np.arange(-4,5)/T).astype(np.float32)
def sweep():
    for mk in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        delta=(s*nn).astype(np.int8)
        r=rng.random(s.shape,dtype=np.float32)
        acc=(delta<=0)|(r<pt[delta+4])
        s[acc&mk]*=-1
t0=time.time()
for _ in range(NEQ): sweep()
nb=(NSW)//BL
S2=np.zeros((nb,R)); S4=np.zeros((nb,R)); S1=np.zeros((nb,R))
for b in range(nb):
    acc2=np.zeros(R); acc4=np.zeros(R); acc1=np.zeros(R)
    for k in range(BL):
        sweep()
        m=s.sum(axis=(1,2))/(N*N)
        acc2+=m*m; acc4+=m**4; acc1+=np.abs(m)
    S2[b]=acc2/BL; S4[b]=acc4/BL; S1[b]=acc1/BL
np.savez(f'/work/T{T}.npz',S2=S2,S4=S4,S1=S1,T=T,R=R,NEQ=NEQ,BL=BL,seed=seed)
m2=S2.mean(); m4=S4.mean()
print("time",time.time()-t0)
print("U pooled",1-m4/(3*m2*m2))
print("block U (mean over replicas, by block):",np.round([1-S4[b].mean()/(3*S2[b].mean()**2) for b in range(nb)],4))
