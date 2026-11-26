import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
N=16
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        s=np.where(acc&masks[c],-s,s).astype(np.int8)
    return s
T=2.3; R=1000; seed=12345
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
t0=time.time()
for _ in range(2000): s=sweep(s,T,rng)
ms=[];
for k in range(10000):
    s=sweep(s,T,rng)
    if k%5==0:
        m=s.reshape(R,-1).mean(1,dtype=np.float64)
        ms.append(m)
ms=np.array(ms)
m2=ms**2; m4=ms**4
nb=ms.shape[0]//200
for b in range(nb):
    a2=m2[b*200:(b+1)*200].mean(); a4=m4[b*200:(b+1)*200].mean()
    print(b, round(1-a4/(3*a2*a2),4))
S2=m2.mean(0); S4=m4.mean(0)
U=1-S4.mean()/(3*S2.mean()**2)
tot2=S2.sum(); tot4=S4.sum()
Uj=np.array([1-((tot4-S4[i])/(R-1))/(3*((tot2-S2[i])/(R-1))**2) for i in range(R)])
err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
print("U=",U,"err=",err,"time",time.time()-t0)
np.save('/work/pilot_T2.3.npy', np.stack([S2,S4]))