
import numpy as np, time, json
N=16; R=1000; T=1.9; seed=1001; NEQ=3000; NMEAS=20000
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
I,J=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((I+J)%2==c) for c in (0,1)]
def sweep(s,T):
    for c in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(rng.random(s.shape)<p)&masks[c]
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t0=time.time()
for _ in range(NEQ): s=sweep(s,T)
S2=np.zeros(R); S4=np.zeros(R); cnt=0; mabs_mean=0.0
half_check=[]
for k in range(NMEAS):
    s=sweep(s,T)
    m=s.reshape(R,-1).mean(axis=1,dtype=np.float64)
    m2=m*m
    S2+=m2; S4+=m2*m2; cnt+=1
    if k==NMEAS//2-1: S2a=S2.copy(); S4a=S4.copy()
    if time.time()-t0>540: break
np.savez(f"/work/res_N{N}_T{T}.npz",S2=S2,S4=S4,cnt=cnt,S2a=S2a,S4a=S4a,n_half=NMEAS//2,seed=seed,NEQ=NEQ)
print("done cnt",cnt,"time",time.time()-t0)
