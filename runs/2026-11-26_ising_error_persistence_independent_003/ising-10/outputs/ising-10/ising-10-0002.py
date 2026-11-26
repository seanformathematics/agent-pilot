import numpy as np, time
N=16; R=400; T=2.3
rng=np.random.default_rng(12345)
S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(S,T):
    for m in masks:
        h=np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2)
        dE=2*S*h  # int8 ok: max 8
        acc=(dE<=0)|(rng.random(S.shape)<np.exp(-np.maximum(dE,0)/T))
        S[:]=np.where(m&acc,-S,S)
    return S
t=time.time()
for _ in range(50): sweep(S,T)
print('time per sweep (R=400):',(time.time()-t)/50)
nsw=1500
m=np.empty((nsw,R))
t=time.time()
for k in range(nsw):
    sweep(S,T)
    m[k]=S.reshape(R,-1).mean(1)
print('time',time.time()-t)
m2=m**2; m4=m**4
def tau(x):
    x=x-x.mean(); n=len(x)
    c=np.fft.irfft(np.abs(np.fft.rfft(x,2*n))**2)[:n]; c/=c[0]
    s=np.where(c<0)[0]; return 1+2*c[1:s[0]].sum()
pm2=m2[300:].mean(0); print('ac m2 approx tau per rep', np.mean([tau(m2[300:,r]) for r in range(0,R,40)]))
print('ac m4', np.mean([tau(m4[300:,r]) for r in range(0,R,40)]))
Mm2=m2[300:].mean(); Mm4=m4[300:].mean()
print('U',1-Mm4/(3*Mm2**2))