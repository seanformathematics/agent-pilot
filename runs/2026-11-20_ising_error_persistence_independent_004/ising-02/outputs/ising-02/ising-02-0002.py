
import numpy as np, time, json, sys
N=16; R=800; burn=3000; meas=25000
T=1.9; seed=1001
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
maskA=((ii+jj)%2==0); maskB=~maskA
def sweep(s,T):
    for mask in (maskA,maskB):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        p=np.exp(-dE.clip(min=0)/T)
        acc=(rng.random(s.shape)<p)&mask
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t0=time.time()
for _ in range(burn): s=sweep(s,T)
S2=np.zeros(R); S4=np.zeros(R)
blk=[]
bm2=np.zeros(R); cnt=0
for k in range(meas):
    s=sweep(s,T)
    m=s.sum(axis=(1,2))/(N*N)
    m2=m*m
    S2+=m2; S4+=m2*m2
    bm2+=m2; cnt+=1
    if cnt==1000:
        blk.append(float(bm2.mean()/cnt)); bm2[:]=0; cnt=0
    if time.time()-t0>540:
        print("stopped early at",k); break
n=k+1
S2/=n; S4/=n
U_all = 1-S4.mean()/(3*S2.mean()**2)
np.savez(f"/work/T{T}_N{N}.npz",S2=S2,S4=S4,n=n)
print("T",T,"sweeps",n,"U",U_all,"blocks m2",np.round(blk,4),"time",time.time()-t0)
