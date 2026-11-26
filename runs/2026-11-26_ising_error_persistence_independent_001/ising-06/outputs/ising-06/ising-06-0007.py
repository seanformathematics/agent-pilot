import numpy as np
N=4; T=2.4
# exact enumeration
states=np.array([[1 if (x>>k)&1 else -1 for k in range(N*N)] for x in range(2**(N*N))],dtype=np.int64).reshape(-1,N,N)
E=-(states*np.roll(states,1,1)).sum((1,2))-(states*np.roll(states,1,2)).sum((1,2))
w=np.exp(-E/T); Z=w.sum()
e_ex=(w*E).sum()/Z/(N*N)
m=states.mean((1,2))
m2=(w*m**2).sum()/Z; m4=(w*m**4).sum()/Z
print("exact e/site",e_ex,"U",1-m4/(3*m2**2))
# simulation with same code
i,j=np.indices((N,N)); masks=[((i+j)%2==p) for p in (0,1)]
R=4000; rng=np.random.default_rng(5)
beta=np.float32(1/T)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
def sweep(s):
    for mk in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nb
        acc=np.exp(-beta*np.maximum(dE,0).astype(np.float32))
        r=rng.random(s.shape,dtype=np.float32)
        s[(r<acc)&mk]*=-1
es=[];m2s=[];m4s=[]
for k in range(3000):
    sweep(s)
    if k>=500 and k%5==0:
        mm=s.reshape(R,-1).mean(1)
        es.append(-(np.roll(s,1,1)*s+np.roll(s,1,2)*s).reshape(R,-1).mean(1).mean()/2)
        m2s.append((mm**2).mean()); m4s.append((mm**4).mean())
m2s=np.mean(m2s); m4s=np.mean(m4s)
print("sim e/site",np.mean(es),"U",1-m4s/(3*m2s**2))