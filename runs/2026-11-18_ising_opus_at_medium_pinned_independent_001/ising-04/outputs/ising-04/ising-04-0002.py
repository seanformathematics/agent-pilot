
import numpy as np, time
Ts=np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
N=16;R=128;nT=len(Ts)
rng=np.random.default_rng(2024)
s=np.ones((nT,R,N,N),dtype=np.int8)
ii,jj=np.indices((N,N));masks=[((ii+jj)%2==c) for c in (0,1)]
# lookup: dE=2*s*nb in {-8,-4,0,4,8} -> index (dE+8)//4
tab=np.minimum(1,np.exp(-np.array([-8,-4,0,4,8])[None,:]/Ts[:,None]))  # nT x5
tabb=tab[:,None,None,None,:]
def sweep(s):
    for m in masks:
        nb=(np.roll(s,1,2)+np.roll(s,-1,2)+np.roll(s,1,3)+np.roll(s,-1,3))
        idx=((s*nb+4)//2).astype(np.intp)  # s*nb in -4..4 -> 0..4
        p=np.take_along_axis(np.broadcast_to(tabb,(nT,R,1,1,5)),idx[...,None],-1)[...,0] if False else tab[np.arange(nT)[:,None,None,None],idx]
        flip=(rng.random(s.shape)<p)&m
        s=np.where(flip,-s,s)
    return s
t0=time.time()
for _ in range(3000): s=sweep(s)
print('equil',time.time()-t0)
M2=np.zeros((nT,R));M4=np.zeros((nT,R));n=0
# also block series for autocorr check: store per-block means
blocks=[]
while time.time()-t0<540:
    b2=np.zeros((nT,R));b4=np.zeros((nT,R))
    for _ in range(500):
        s=sweep(s)
        m=s.mean(axis=(2,3),dtype=np.float64)
        b2+=m**2;b4+=m**4
    M2+=b2;M4+=b4;n+=500;blocks.append((b2/500,b4/500))
np.savez('/work/run1.npz',M2=M2,M4=M4,n=n,s=s)
print(n)
m2=M2/n;m4=M4/n
U=1-m4.mean(1)/(3*m2.mean(1)**2)
# jackknife over replicas
Uj=np.array([1-(m4.sum(1)-m4[:,r])/(R-1)/(3*((m2.sum(1)-m2[:,r])/(R-1))**2) for r in range(R)])
err=np.sqrt((R-1)/R*((Uj-Uj.mean(0))**2).sum(0))
for T,u,e in zip(Ts,U,err): print(T,round(u,5),round(e,5))
