import numpy as np, time, os
N=16
Ts=np.array([1.9,2.0,2.1,2.2,2.3,2.4])
K=2000
R=len(Ts)*K
i,j=np.indices((N,N))
masks=[((i+j)%2==p) for p in (0,1)]
rng=np.random.default_rng(777001)
T=np.repeat(Ts,K)
beta=(1.0/T).astype(np.float32).reshape(-1,1,1)
s=np.load('/work/state_s.npy')
def sweep(s):
    for m in masks:
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE = 2*s*nb
        acc = np.exp(-beta*np.maximum(dE,0).astype(np.float32))
        r = rng.random(s.shape,dtype=np.float32)
        flip = (r<acc) & m
        s[flip] *= -1
NSW=4000; BLK=100
nblk=NSW//BLK
M2=np.zeros((nblk,R)); M4=np.zeros((nblk,R))
t0=time.time()
for b in range(nblk):
    acc2=np.zeros(R); acc4=np.zeros(R)
    for k in range(BLK):
        sweep(s)
        mm=s.reshape(R,-1).mean(1)
        m2=mm*mm
        acc2+=m2; acc4+=m2*m2
    M2[b]=acc2/BLK; M4[b]=acc4/BLK
np.save('/work/state_s.npy', s)
np.savez('/work/meas_1.npz', M2=M2, M4=M4)
print("done", time.time()-t0)
U=1-(M4.mean(0)/(3*M2.mean(0)**2))
for t in range(len(Ts)):
    print(Ts[t], np.mean(1-M4[:,t*K:(t+1)*K].mean(0)/(3*M2[:,t*K:(t+1)*K].mean(0)**2)))
