
import numpy as np, time
Ts=np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
N=16;R=64;K=len(Ts)
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(K,R,N,N))
beta=(1/Ts)[:,None,None,None]
ii,jj=np.indices((N,N));mask=[((ii+jj)%2==c) for c in (0,1)]
# precompute acceptance for dE=4,8
a4=np.exp(-4*beta);a8=np.exp(-8*beta)
def sweep():
    for c in (0,1):
        nb=np.roll(s,1,2)+np.roll(s,-1,2)+np.roll(s,1,3)+np.roll(s,-1,3)
        h=s*nb  # dE=2*h
        p=np.where(h<=0,1.0,np.where(h==2,a4,a8))
        flip=(rng.random(s.shape)<p)&mask[c]
        s[flip]*=-1
t=time.time()
for _ in range(1000): sweep()
print(time.time()-t)
S2=np.zeros((K,R));S4=np.zeros((K,R));n=0
t=time.time()
while time.time()-t<420:
    for _ in range(4): sweep()
    m=s.reshape(K,R,-1).sum(2)/N**2
    S2+=m**2;S4+=m**4;n+=1
print(n)
np.save('/work/S.npy',np.stack([S2,S4])/n)
M2=S2/n;M4=S4/n
U=1-M4.mean(1)/(3*M2.mean(1)**2)
# jackknife over replicas
Uj=np.array([[1-np.delete(M4[k],r).mean()/(3*np.delete(M2[k],r).mean()**2) for r in range(R)] for k in range(K)])
err=np.sqrt((R-1)/R*((Uj-Uj.mean(1,keepdims=True))**2).sum(1))
for T,u,e in zip(Ts,U,err):print(T,u,e)
