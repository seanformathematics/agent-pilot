
import numpy as np, time, json, sys
N=16; R=512; T=float(sys.argv[1]) if len(sys.argv)>1 else 1.9
seed=1000+int(round(T*100))
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
        s[acc&m]*=-1
    return s
NEQ=3000; NMEAS=20000; BLK=200
t=time.time()
for k in range(NEQ): sweep(s,T)
nb_blocks=NMEAS//BLK
m2b=np.zeros((nb_blocks,R)); m4b=np.zeros((nb_blocks,R))
for k in range(NMEAS):
    sweep(s,T)
    m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
    b=k//BLK
    m2b[b]+=m*m; m4b[b]+=m**4
m2b/=BLK; m4b/=BLK
np.savez(f"/work/T{T:.2f}.npz",m2b=m2b,m4b=m4b)
print("T",T,"time",time.time()-t)
